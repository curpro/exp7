import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt  # [新增] 用于画图
from collections import deque
from scipy.signal import savgol_filter

# 请确保路径与您项目结构一致
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced

# ================= 配置 =================
# [修改] 指向您的 F16 测试文件路径
TEST_DATA_PATH = r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv'

MODEL_PATH = 'imm_param_net.pth'
SCALER_PATH = 'scaler_params.json'

NUM_MC_TRIALS = 50 # 20-50

WINDOW_SIZE = 90
DT = 1 / 30  # 假设采样率为 30Hz，如果CSV里有时间戳，最好通过时间戳计算
OPTIMIZE_INTERVAL = 20
SAVGOL_WINDOW = 25 # [新增] 与训练一致
SAVGOL_POLY = 2


# === [核心修改] 模型定义必须与训练代码完全一致 ===
class ParamPredictorMLP(nn.Module):
    def __init__(self, seq_len=90, input_dim=9):
        super(ParamPredictorMLP, self).__init__()
        # 输入维度 = 时间步长 * 特征数 (例如 90 * 9 = 810)
        self.input_flat_dim = seq_len * input_dim
        # 定义 MLP 网络结构：输入 -> 隐层 -> 输出
        # 这里设计了一个 3 层网络 (810 -> ->128 -> 32-> 9)
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.BatchNorm1d(128),  # 加速收敛，防止过拟合
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            # 防止过拟合
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(32, 9)  # 输出 9 个值，对应 3x3 矩阵
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, 90, 9)
        b, s, f = x.shape
        # [关键] 将时间序列展平：(batch, 810)
        x = x.reshape(b, -1)
        logits = self.net(x)
        # 后续处理保持不变，与原代码兼容
        logits = logits.view(-1, 3, 3)
        temperature = 2.0
        return torch.log_softmax(logits / temperature, dim=2)


# ================= 特征提取函数 (与 Step1 完全一致) =================
def calculate_derivatives(pos_data, dt):
    """
    [修改] 使用 scipy.signal.savgol_filter 计算速度和加速度。
    逻辑完全复用 step1_generate_data.py 中的代码。
    """
    # 如果数据太短无法滤波，回退到原来的逻辑（防止报错）
    if len(pos_data) < SAVGOL_WINDOW:
        vel = np.zeros_like(pos_data)
        vel[1:] = (pos_data[1:] - pos_data[:-1]) / dt
        vel[0] = vel[1]

        acc = np.zeros_like(pos_data)
        acc[1:] = (vel[1:] - vel[:-1]) / dt
        acc[0] = acc[1]
        return vel, acc

    # 使用 scipy 的 savgol_filter
    # deriv=1 算一阶导(速度), deriv=2 算二阶导(加速度)
    vel = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=1, delta=dt, axis=0)
    acc = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=2, delta=dt, axis=0)

    return vel, acc


def load_test_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到测试文件: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError(f"CSV 文件缺少必要的位置列 x, y, z")

    pos_gt = df[['x', 'y', 'z']].values

    # === [修改部分开始] ===
    # 1. 先算一遍导数 (用于获取加速度，或者作为速度的备选)
    calc_vel, calc_acc = calculate_derivatives(pos_gt, DT)

    # 2. 尝试从 CSV 读取速度真值 (与 main_nn_inference.py 保持一致)
    vel_cols = ['vx', 'vy', 'vz']
    if all(col in df.columns for col in vel_cols):
        print("  > [Info] 成功读取 CSV 中的真值速度 (用于评估)")
        vel_gt = df[vel_cols].values
        # CSV 通常没有加速度，所以加速度依然用算出来的
        acc_gt = calc_acc
    else:
        print("  > [Info] CSV 中未找到速度列，使用计算导数作为真值")
        vel_gt = calc_vel
        acc_gt = calc_acc
    # === [修改部分结束] ===

    print(f"  > 数据加载完成。点数: {len(pos_gt)}")
    return pos_gt, vel_gt, acc_gt


def run_comparison_simulation(noise_std, gt_pos_data, gt_vel_data, gt_acc_data, model, mean, std, device, nn_seed, fix_seed):
    sim_steps = len(gt_pos_data)

    # 1. 噪声生成 (保持不变)
    np.random.seed(nn_seed)
    noise_matrix_nn = np.random.randn(3, sim_steps) * noise_std
    meas_pos_nn = gt_pos_data + noise_matrix_nn.T

    np.random.seed(fix_seed)
    noise_matrix_fix = np.random.randn(3, sim_steps) * noise_std
    meas_pos_fix = gt_pos_data + noise_matrix_fix.T

    np.random.seed(414)

    # 2. 初始化参数
    fixed_trans_prob = np.array([
        [0.81388511, 0.18511489, 0.001],
        [0.989, 0.01, 0.001],
        [0.01, 0.01, 0.98]
    ])

    # === [核心修正 1: 状态初始化对齐] ===
    init_state = np.zeros(9)
    # (1) 位置：从 CSV 获取
    init_state[[0, 3, 6]] = gt_pos_data[0]
    # (2) 速度：从 CSV 获取 (与您指出的 main_nn 代码一致)
    init_state[[1, 4, 7]] = gt_vel_data[0]
    # (3) 加速度：main_nn 中没赋值(即为0)，所以这里必须强制设为 0，不能用 gt_acc_data！
    init_state[[2, 5, 8]] = 0.0

    # === [核心修正 2: 协方差矩阵 P0 对齐] ===
    # 原代码 eye(9)*100 太粗糙，必须换回 main_nn 的精细配置
    init_cov_diag = np.zeros(9)
    init_cov_diag[[0, 3, 6]] = 100.0  # Pos
    init_cov_diag[[1, 4, 7]] = 25.0   # Vel
    init_cov_diag[[2, 5, 8]] = 10.0   # Acc
    init_cov = np.diag(init_cov_diag)

    current_R = np.eye(3) * (noise_std ** 2)

    # 实例化滤波器
    imm_adapt = IMMFilterEnhanced(fixed_trans_prob, init_state, init_cov, r_cov=current_R)
    imm_fixed = IMMFilterEnhanced(fixed_trans_prob, init_state, init_cov, r_cov=current_R)

    # 3. 循环变量
    pos_buffer = deque(maxlen=WINDOW_SIZE)
    last_pred_params = None
    alpha_smooth = 0.9

    err_sq_sum_adapt = np.zeros(3)
    err_sq_sum_fixed = np.zeros(3)
    valid_steps = 0

    # 4. 仿真循环
    for k in range(sim_steps):
        z_k_nn = meas_pos_nn[k]
        z_k_fix = meas_pos_fix[k]

        # --- NN 推理 ---
        if len(pos_buffer) == WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
            pos_seq = np.array(pos_buffer)
            vel_seq, acc_seq = calculate_derivatives(pos_seq, DT)
            raw_features = np.hstack([pos_seq - pos_seq[-1], vel_seq, acc_seq])
            norm_features = (raw_features - mean) / std

            inp = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.exp(model(inp)).cpu().numpy()[0]

            if last_pred_params is not None:
                pred = alpha_smooth * pred + (1 - alpha_smooth) * last_pred_params
            last_pred_params = pred

            new_mtx = np.clip(pred, 1e-6, 1.0)
            new_mtx = new_mtx / np.sum(new_mtx, axis=1, keepdims=True)
            imm_adapt.set_transition_matrix(new_mtx)

        # --- 滤波器更新 ---
        est_adapt, _ = imm_adapt.update(z_k_nn, DT)
        est_fixed, _ = imm_fixed.update(z_k_fix, DT)

        pos_buffer.append(z_k_nn)

        # --- 误差统计 (跳过前 90 帧) ---
        if k >= WINDOW_SIZE:
            err_sq_sum_adapt[0] += np.sum((est_adapt[[0, 3, 6]] - gt_pos_data[k]) ** 2)
            err_sq_sum_adapt[1] += np.sum((est_adapt[[1, 4, 7]] - gt_vel_data[k]) ** 2)
            # 统计时可以用 gt_acc_data 做参考，这不影响滤波器运行
            err_sq_sum_adapt[2] += np.sum((est_adapt[[2, 5, 8]] - gt_acc_data[k]) ** 2)

            err_sq_sum_fixed[0] += np.sum((est_fixed[[0, 3, 6]] - gt_pos_data[k]) ** 2)
            err_sq_sum_fixed[1] += np.sum((est_fixed[[1, 4, 7]] - gt_vel_data[k]) ** 2)
            err_sq_sum_fixed[2] += np.sum((est_fixed[[2, 5, 8]] - gt_acc_data[k]) ** 2)
            valid_steps += 1

    return np.sqrt(err_sq_sum_adapt / valid_steps), np.sqrt(err_sq_sum_fixed / valid_steps)


def calculate_snr_db(signal_data, noise_std):
    """
    计算轨迹的信噪比 (SNR)
    Formula: SNR_dB = 10 * log10(P_signal / P_noise)
    """
    # 1. 计算信号功率 (P_signal)
    # 对于位置轨迹，我们通常关注其相对于均值的变化量，或者运动的剧烈程度
    # 这里我们计算信号的方差作为功率估计 (Variance ~ Power of AC component)
    # axis=0 求每个轴(x,y,z)的方差，然后求和得到总功率
    signal_power = np.var(signal_data, axis=0).sum()

    # 2. 计算噪声功率 (P_noise)
    # 噪声是标准差为 noise_std 的高斯白噪声
    # P_noise = sigma^2 * 3 (因为是 x,y,z 三个轴)
    noise_power = (noise_std ** 2) * 3

    # 3. 计算 SNR (dB)
    if noise_power < 1e-9: return 100.0  # 避免除零
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def main_inference():
    # === [核心设置] 定义你的“主角”种子 ===
    GLOBAL_SEED = 414
    FIXED_TARGET_SEED = 42

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)  # 全局初始化
    if torch.cuda.is_available():
        torch.cuda.manual_seed(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        gt_pos, gt_vel, gt_acc = load_test_data(TEST_DATA_PATH)
        model = ParamPredictorMLP(input_dim=9).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        with open(SCALER_PATH, 'r') as f:
            scaler = json.load(f)
        mean, std = np.array(scaler['mean'], np.float32), np.array(scaler['std'], np.float32)
        print(">>> 资源加载完毕...")
    except Exception as e:
        print(f"Error: {e}")
        return

    noise_levels = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]

    # 1. 存储 Seed 42 的结果 (用于画实线，也是阴影的中心骨架)
    results_main_adapt = []
    results_main_fixed = []

    # 2. 存储 Monte Carlo 的统计量 (只用它的 std 来决定阴影胖瘦)
    mc_stds_adapt = []
    mc_stds_fixed = []

    snr_list = []

    print(f"\n>>> 开始混合模式仿真：")
    print(f"    - 实线: 典型运行 (Seed={FIXED_TARGET_SEED})")
    print(f"    - 阴影: 基于 Seed 42 的波动范围 (Seed 42 ± MC Std)")

    for i, lvl in enumerate(noise_levels):
        snr_list.append(calculate_snr_db(gt_pos, lvl))
        print(f"  > Noise Level {lvl} (SNR: {snr_list[-1]:.1f}dB)...")

        # --- A. 跑 Seed 42 (你的“真理”线) ---
        ra_main, rf_main = run_comparison_simulation(
            lvl, gt_pos, gt_vel, gt_acc, model, mean, std, device,
            nn_seed=GLOBAL_SEED,  # 414
            fix_seed=FIXED_TARGET_SEED  # 42
        )
        results_main_adapt.append(ra_main)
        results_main_fixed.append(rf_main)

        # --- B. 跑 Monte Carlo Loop (只为了算 Std) ---
        trials_adapt = []
        trials_fixed = []
        for t in range(NUM_MC_TRIALS):
            # 使用随机种子 (避开 MAIN_SEED)
            current_nn_seed = GLOBAL_SEED + 1000 + t * 999
            current_fix_seed = FIXED_TARGET_SEED + 1000 + t * 999
            ra, rf = run_comparison_simulation(
                lvl, gt_pos, gt_vel, gt_acc, model, mean, std, device,
                nn_seed=current_nn_seed,
                fix_seed=current_fix_seed
            )
            trials_adapt.append(ra)
            trials_fixed.append(rf)

        # 计算标准差 (Std)
        mc_stds_adapt.append(np.std(trials_adapt, axis=0))
        mc_stds_fixed.append(np.std(trials_fixed, axis=0))

    # --- 数据转换 ---
    # Seed 42 数据 (实线 + 阴影中心)
    res_42_adapt_arr = np.array(results_main_adapt)  # 这里的命名保留原代码习惯，方便你后面不用改太多
    res_42_fixed_arr = np.array(results_main_fixed)

    # 统计数据 (阴影宽度)
    std_adapt_arr = np.array(mc_stds_adapt)
    std_fixed_arr = np.array(mc_stds_fixed)

    print("\n" + "=" * 90)
    print(f"{'>>> 详细 RMSE 数值统计表 (基于 Seed 42) <<<':^90}")
    print("=" * 90)
    print(
        f"{'Noise':<6} | {'SNR(dB)':<8} | {'Type':<5} | {'Fixed RMSE':<12} | {'NN-IMM RMSE':<12} | {'Improvement':<10}")
    print("-" * 90)

    for idx, lvl in enumerate(noise_levels):
        snr = snr_list[idx]
        # 遍历三个维度: 0=Pos, 1=Vel, 2=Acc
        for dim, name in enumerate(['Pos', 'Vel', 'Acc']):
            val_fix = res_42_fixed_arr[idx, dim]
            val_nn = res_42_adapt_arr[idx, dim]
            imp = (val_fix - val_nn) / val_fix * 100

            # 为了美观，只在每组的第一行显示 Noise 和 SNR
            d_lvl = str(lvl) if dim == 0 else ""
            d_snr = f"{snr:.1f}" if dim == 0 else ""

            print(f"{d_lvl:<6} | {d_snr:<8} | {name:<5} | {val_fix:<12.4f} | {val_nn:<12.4f} | {imp:>9.2f}%")
        print("-" * 90)
    print("\n")

    # ================= 绘图逻辑 (Seed 42 ± Std) =================
    titles = ['Position RMSE (seed=42; MonteCarlo μ±σ) ', 'Velocity RMSE (seed=42; MonteCarlo μ±σ)', 'Acceleration RMSE (seed=42; MonteCarlo μ±σ)']
    ylabels = ['RMSE (m)', 'RMSE (m/s)', 'RMSE (m/s²)']

    color_nn = '#D62728'  # 红色
    color_fix = '#1F77B4'  # 蓝色

    for i in range(3):
        fig, ax1 = plt.subplots(figsize=(9, 7))

        # --- 1. 画 Fixed IMM ---
        # A. 实线：Seed 42
        ax1.plot(noise_levels, res_42_fixed_arr[:, i], marker='^', linestyle='--', color=color_fix,
                 label='BO-IMM (Seed 42)', linewidth=1.5)

        # B. 阴影：【关键修改】 以 Seed 42 为中心，上下加减 Std
        #    这样线一定在阴影正中间
        ax1.fill_between(noise_levels,
                         res_42_fixed_arr[:, i] - std_fixed_arr[:, i],
                         res_42_fixed_arr[:, i] + std_fixed_arr[:, i],
                         color=color_fix, alpha=0.15, linewidth=0,
                         label='BO-IMM (± σ)')

        # --- 2. 画 NN-IMM ---
        # A. 实线：Seed 42
        ax1.plot(noise_levels, res_42_adapt_arr[:, i], marker='o', linestyle='-', color=color_nn,
                 label='NN-IMM (Seed 42)', linewidth=2.0)

        # B. 阴影：【关键修改】 以 Seed 42 为中心，上下加减 Std
        ax1.fill_between(noise_levels,
                         res_42_adapt_arr[:, i] - std_adapt_arr[:, i],
                         res_42_adapt_arr[:, i] + std_adapt_arr[:, i],
                         color=color_nn, alpha=0.2, linewidth=0,
                         label='NN-IMM (± σ)')

        # --- 3. 提升率标注 (基于 Seed 42) ---
        improv_pct = (res_42_fixed_arr[:, i] - res_42_adapt_arr[:, i]) / res_42_fixed_arr[:, i] * 100

        for idx, lvl in enumerate(noise_levels):
            val_nn = res_42_adapt_arr[idx, i]
            val_fix = res_42_fixed_arr[idx, i]
            imp = improv_pct[idx]

            if imp > 0.5:
                # 文字稍微避让一下
                ax1.text(lvl, val_nn - (val_fix - val_nn) * 0.15,
                         f'↓{imp:.1f}%',
                         ha='center', va='top', fontsize=9, color='darkred', fontweight='bold')

        # --- 装饰 ---
        ax1.set_xlabel('Measurement Noise σ (m)', fontsize=12)
        ax1.set_ylabel(ylabels[i], fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # SNR 轴
        # 1. 筛选出 5, 10, 15... 的刻度位置和对应的 SNR
        target_indices = [idx for idx, val in enumerate(noise_levels) if val % 5 == 0]
        target_ticks = [noise_levels[i] for i in target_indices]
        target_snrs = [f"{snr_list[i]:.1f}" for i in target_indices]

        # 2. 强制下轴 (Noise) 只显示 5, 10, 15...
        ax1.set_xticks(target_ticks)

        # 3. 设置上轴 (SNR) 与下轴完全对齐
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(target_ticks)  # 位置和下轴一致
        ax2.set_xticklabels(target_snrs)  # 标签显示对应的 SNR
        ax2.set_xlabel('SNR (dB)', fontsize=11)
        ax2.tick_params(axis='x')

        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        plt.title(f"{titles[i]}", fontsize=13,pad=15)
        plt.tight_layout()
        plt.show()

    print("\n>>> 绘图完成。")
    print(">>> 策略：以 Seed 42 为骨架，蒙特卡洛 Std 为皮肉。")
    print(">>> 效果：实线完美居中，且保留了统计学的波动范围信息。")

if __name__ == '__main__':
    main_inference()