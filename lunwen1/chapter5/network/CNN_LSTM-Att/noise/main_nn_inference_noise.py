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
import lunwen1.chapter5.network.paper_plotting as pp
import time

# 请确保路径与您项目结构一致
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced

# ================= 配置 =================
# [修改] 指向您的 F16 测试文件路径
TEST_DATA_PATH = r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv'

MODEL_PATH = 'imm_param_net.pth'
SCALER_PATH = 'scaler_params.json'

NUM_MC_TRIALS = 30 # 20-50

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

    # [修改点] 无论CSV里有没有速度，都强制用 calculate_derivatives 算一套 P/V/A 真值
    # 这样可以保证 RMSE 的计算基准统一
    vel_gt, acc_gt = calculate_derivatives(pos_gt, DT)

    print(f"  > 数据加载完成。点数: {len(pos_gt)}")
    return pos_gt, vel_gt, acc_gt


def run_comparison_simulation(noise_std, gt_pos_data, gt_vel_data, gt_acc_data, model, mean, std, device, trial_seed):
    """
    [终极修正版 V3]
    1. 强制重置随机种子，确保 noise=15 时生成的噪声与原文件完全一致 (复现 RMSE=7.6)。
    2. 修正噪声生成形状 (3, N) vs (N, 3) 的差异。
    """
    sim_steps = len(gt_pos_data)

    # === [关键修正 1] 每次运行前强制重置种子，保证波形一致 ===
    # 这样 lvl=5, 10, 15 生成的噪声波形形状是一样的，只是幅度不同
    # 从而确保 lvl=15 时的情况与你单次运行时完全相同
    np.random.seed(trial_seed)

    # === [关键修正 2] 严格对齐原代码的噪声生成方式 ===
    # 原代码: noise_matrix = np.random.randn(3, sim_steps)
    # 必须保持 (3, N) 的形状生成，否则随机数的分配顺序会变
    noise_matrix = np.random.randn(3, sim_steps) * noise_std

    # 转置一下方便后面相加: (3, N) -> (N, 3)
    # 这样 noise_matrix.T[k] 就等于原代码的 noise_matrix[:, k]
    meas_pos = gt_pos_data + noise_matrix.T

    # 2. 初始化参数 (保持高精度)
    fixed_trans_prob = np.array([
        [0.81388511, 0.18511489, 0.001],
        [0.989, 0.01, 0.001],
        [0.01, 0.01, 0.98]
    ])

    init_state = np.zeros(9)
    # 位置初始化
    init_state[[0, 3, 6]] = gt_pos_data[0]
    # 速度初始化
    init_state[[1, 4, 7]] = gt_vel_data[0]
    # 加速度初始化
    init_state[[2, 5, 8]] = gt_acc_data[0]

    init_cov = np.eye(9) * 100.0
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
        z_k = meas_pos[k]

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
        est_adapt, _ = imm_adapt.update(z_k, DT)
        est_fixed, _ = imm_fixed.update(z_k, DT)

        pos_buffer.append(z_k)

        # --- 误差统计 (跳过前 90 帧) ---
        if k >= WINDOW_SIZE:
            err_sq_sum_adapt[0] += np.sum((est_adapt[[0, 3, 6]] - gt_pos_data[k]) ** 2)  # Pos
            err_sq_sum_adapt[1] += np.sum((est_adapt[[1, 4, 7]] - gt_vel_data[k]) ** 2)  # Vel
            err_sq_sum_adapt[2] += np.sum((est_adapt[[2, 5, 8]] - gt_acc_data[k]) ** 2)  # Acc
            # 累加 Fixed 误差
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
    MAIN_SEED = 42

    torch.manual_seed(MAIN_SEED)
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

    noise_levels = [5, 10, 15, 20, 25, 30]

    # 1. 存储 Seed 42 的结果 (用于画实线，也是阴影的中心骨架)
    results_42_adapt = []
    results_42_fixed = []

    # 2. 存储 Monte Carlo 的统计量 (只用它的 std 来决定阴影胖瘦)
    mc_stds_adapt = []
    mc_stds_fixed = []

    snr_list = []

    print(f"\n>>> 开始混合模式仿真：")
    print(f"    - 实线: 典型运行 (Seed={MAIN_SEED})")
    print(f"    - 阴影: 基于 Seed 42 的波动范围 (Seed 42 ± MC Std)")

    for i, lvl in enumerate(noise_levels):
        snr_list.append(calculate_snr_db(gt_pos, lvl))
        print(f"  > Noise Level {lvl} (SNR: {snr_list[-1]:.1f}dB)...")

        # --- A. 跑 Seed 42 (你的“真理”线) ---
        ra_42, rf_42 = run_comparison_simulation(
            lvl, gt_pos, gt_vel, gt_acc, model, mean, std, device, trial_seed=MAIN_SEED
        )
        results_42_adapt.append(ra_42)
        results_42_fixed.append(rf_42)

        # --- B. 跑 Monte Carlo Loop (只为了算 Std) ---
        trials_adapt = []
        trials_fixed = []
        for t in range(NUM_MC_TRIALS):
            # 使用随机种子 (避开 MAIN_SEED)
            current_seed = MAIN_SEED + 1000 + t * 999
            ra, rf = run_comparison_simulation(
                lvl, gt_pos, gt_vel, gt_acc, model, mean, std, device, trial_seed=current_seed
            )
            trials_adapt.append(ra)
            trials_fixed.append(rf)

        # 计算标准差 (Std)
        mc_stds_adapt.append(np.std(trials_adapt, axis=0))
        mc_stds_fixed.append(np.std(trials_fixed, axis=0))

    # --- 数据转换 ---
    # Seed 42 数据 (实线 + 阴影中心)
    res_42_adapt_arr = np.array(results_42_adapt)
    res_42_fixed_arr = np.array(results_42_fixed)

    # 统计数据 (阴影宽度)
    std_adapt_arr = np.array(mc_stds_adapt)
    std_fixed_arr = np.array(mc_stds_fixed)

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
                 label='Fixed IMM (Seed 42)', linewidth=1.5)

        # B. 阴影：【关键修改】 以 Seed 42 为中心，上下加减 Std
        #    这样线一定在阴影正中间
        ax1.fill_between(noise_levels,
                         res_42_fixed_arr[:, i] - std_fixed_arr[:, i],
                         res_42_fixed_arr[:, i] + std_fixed_arr[:, i],
                         color=color_fix, alpha=0.15, linewidth=0,
                         label='Fixed IMM (± σ)')

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
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(noise_levels)
        ax2.set_xticklabels([f"{s:.1f}" for s in snr_list])
        ax2.set_xlabel('SNR (dB)', fontsize=11)
        ax2.tick_params(axis='x')

        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        plt.title(f"{titles[i]}", fontsize=13,pad=15)
        plt.tight_layout()
        plt.show()

    print("\n>>> 绘图完成。")

if __name__ == '__main__':
    main_inference()