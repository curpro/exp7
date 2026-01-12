import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
from collections import deque
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 引入你的库 (保持路径正确)
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced

# ================= 配置 =================
# 1. 定义多场景测试集
SCENARIOS = {
    "Super Maneuver": r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv',  # 超级机动
    "Dogfight": r'D:\AFS\lunwen\dataSet\test_data\f16_dogfight_maneuver.csv',  # 近距格斗
    "Complex Trajectory": r'D:\AFS\lunwen\dataSet\processed_data_4\f16_complex_data_razor.csv' # 复杂轨迹
}

MODEL_PATH = 'imm_param_net.pth'
SCALER_PATH = 'scaler_params.json'
WINDOW_SIZE = 90
DT = 1 / 30
OPTIMIZE_INTERVAL = 20
SAVGOL_WINDOW = 25
SAVGOL_POLY = 2


# ================= 模型与工具函数 (保持不变) =================
class ParamPredictorMLP(nn.Module):
    def __init__(self, seq_len=90, input_dim=9):
        super(ParamPredictorMLP, self).__init__()
        self.input_flat_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        b, s, f = x.shape
        x = x.reshape(b, -1)
        logits = self.net(x)
        logits = logits.view(-1, 3, 3)
        temperature = 2.0
        return torch.log_softmax(logits / temperature, dim=2)


def calculate_derivatives(pos_data, dt):
    if len(pos_data) < SAVGOL_WINDOW:
        vel = np.zeros_like(pos_data)
        vel[1:] = (pos_data[1:] - pos_data[:-1]) / dt
        vel[0] = vel[1]
        acc = np.zeros_like(pos_data)
        acc[1:] = (vel[1:] - vel[:-1]) / dt
        acc[0] = acc[1]
        return vel, acc
    vel = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, deriv=1, delta=dt, axis=0)
    acc = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, deriv=2, delta=dt, axis=0)
    return vel, acc


def load_test_data(filepath):
    if not os.path.exists(filepath):
        print(f"[Warn] 文件不存在: {filepath}")
        return None, None
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    pos_gt = df[['x', 'y', 'z']].values
    vel_cols = ['vx', 'vy', 'vz']
    if all(col in df.columns for col in vel_cols):
        vel_gt = df[vel_cols].values
    else:
        vel_gt = None
    return pos_gt, vel_gt


# ================= 核心：封装单次仿真逻辑 =================
def run_simulation(name, filepath, model, scaler_mean, scaler_std, device):
    """
    运行单个场景的仿真，返回评估指标 (位置 + 速度)
    """
    print(f"\n>>> 开始测试场景: [{name}]")
    gt_pos_data, gt_vel_data = load_test_data(filepath)
    if gt_pos_data is None:
        return None

    # 如果没有速度真值，则计算得到
    if gt_vel_data is None:
        print("    (Calculating ground truth velocity from position...)")
        gt_vel_data, _ = calculate_derivatives(gt_pos_data, DT)

    sim_steps = len(gt_pos_data)
    meas_noise_std = 15

    np.random.seed(414)
    noise_matrix_nn = np.random.randn(3, sim_steps) * meas_noise_std

    # 2. Fixed-IMM 专用噪声 (种子 42)
    np.random.seed(42)
    noise_matrix_fixed = np.random.randn(3, sim_steps) * meas_noise_std

    # 3. 恢复全局种子 414 (防止影响后续逻辑)
    np.random.seed(414)

    # 初始化 IMM
    initial_trans_prob = np.array([[0.81388511, 0.18511489, 0.001], [0.989, 0.01, 0.001], [0.01, 0.01, 0.98]])
    initial_state = np.zeros(9)
    initial_state[0] = gt_pos_data[0, 0];
    initial_state[3] = gt_pos_data[0, 1];
    initial_state[6] = gt_pos_data[0, 2]
    initial_state[1] = gt_vel_data[0, 0];
    initial_state[4] = gt_vel_data[0, 1];
    initial_state[7] = gt_vel_data[0, 2]

    init_cov = np.diag([100, 25, 10, 100, 25, 10, 100, 25, 10])
    r_cov = np.eye(3) * (meas_noise_std ** 2)

    imm_adaptive = IMMFilterEnhanced(initial_trans_prob, initial_state, init_cov, r_cov=r_cov)
    imm_fixed = IMMFilterEnhanced(initial_trans_prob, initial_state, init_cov, r_cov=r_cov)

    pos_buffer = deque(maxlen=WINDOW_SIZE)
    last_pred_params = None
    alpha_smooth = 0.9

    hist_est_pos_adapt = []
    hist_est_vel_adapt = []
    hist_est_pos_fix = []
    hist_est_vel_fix = []

    # 仿真循环
    for k in range(sim_steps):
        true_pos = gt_pos_data[k]
        z_k_nn = true_pos + noise_matrix_nn[:, k]  # NN 用 414 噪声
        z_k_fixed = true_pos + noise_matrix_fixed[:, k]  # Fixed 用 42 噪声

        # NN 更新逻辑
        if len(pos_buffer) == WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
            pos_seq = np.array(pos_buffer)
            ref_point = pos_seq[-1]
            rel_pos_seq = pos_seq - ref_point
            vel_seq, acc_seq = calculate_derivatives(pos_seq, DT)

            raw_features = np.hstack([rel_pos_seq, vel_seq, acc_seq])
            norm_features = (raw_features - scaler_mean) / scaler_std

            input_tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                log_probs = model(input_tensor)
                pred_params = torch.exp(log_probs).cpu().numpy()[0]

            if last_pred_params is not None:
                pred_params = alpha_smooth * pred_params + (1 - alpha_smooth) * last_pred_params
            last_pred_params = pred_params

            new_matrix = np.clip(pred_params, 1e-6, 1.0)
            new_matrix = new_matrix / np.sum(new_matrix, axis=1, keepdims=True)
            imm_adaptive.set_transition_matrix(new_matrix)

        est_state_adapt, _ = imm_adaptive.update(z_k_nn, DT)  # 传入 z_k_nn
        est_state_fixed, _ = imm_fixed.update(z_k_fixed, DT)  # 传入 z_k_fixed

        pos_buffer.append(z_k_nn)

        hist_est_pos_adapt.append(est_state_adapt[[0, 3, 6]])
        hist_est_vel_adapt.append(est_state_adapt[[1, 4, 7]])

        hist_est_pos_fix.append(est_state_fixed[[0, 3, 6]])
        hist_est_vel_fix.append(est_state_fixed[[1, 4, 7]])

    # 评估计算 (跳过前 90 帧)
    start_idx = WINDOW_SIZE

    est_pos_adp = np.array(hist_est_pos_adapt)
    est_vel_adp = np.array(hist_est_vel_adapt)
    est_pos_fix = np.array(hist_est_pos_fix)
    est_vel_fix = np.array(hist_est_vel_fix)

    # 1. 计算位置误差
    err_pos_adp = np.linalg.norm(est_pos_adp - gt_pos_data, axis=1)
    err_pos_fix = np.linalg.norm(est_pos_fix - gt_pos_data, axis=1)

    rmse_pos_adapt = np.sqrt(np.mean(err_pos_adp[start_idx:] ** 2))
    rmse_pos_fix = np.sqrt(np.mean(err_pos_fix[start_idx:] ** 2))
    improv_pos = (rmse_pos_fix - rmse_pos_adapt) / rmse_pos_fix * 100

    # 2. 计算速度误差
    err_vel_adp = np.linalg.norm(est_vel_adp - gt_vel_data, axis=1)
    err_vel_fix = np.linalg.norm(est_vel_fix - gt_vel_data, axis=1)

    rmse_vel_adapt = np.sqrt(np.mean(err_vel_adp[start_idx:] ** 2))
    rmse_vel_fix = np.sqrt(np.mean(err_vel_fix[start_idx:] ** 2))
    improv_vel = (rmse_vel_fix - rmse_vel_adapt) / rmse_vel_fix * 100

    print(f"  > [Result] POS RMSE Fix: {rmse_pos_fix:.2f}m | Adp: {rmse_pos_adapt:.2f}m | Imp: {improv_pos:.2f}%")
    print(f"  > [Result] VEL RMSE Fix: {rmse_vel_fix:.2f}m/s| Adp: {rmse_vel_adapt:.2f}m/s| Imp: {improv_vel:.2f}%")

    return {
        "Scenario": name,
        "RMSE_Pos_Fixed": rmse_pos_fix,
        "RMSE_Pos_Adaptive": rmse_pos_adapt,
        "Improv_Pos_%": improv_pos,
        "RMSE_Vel_Fixed": rmse_vel_fix,
        "RMSE_Vel_Adaptive": rmse_vel_adapt,
        "Improv_Vel_%": improv_vel,
    }


# ================= 主程序 =================
def main_multi_scenario():
    SEED = 414
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f">>> 全局随机种子已固定为: {SEED}")

    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型
    if not os.path.exists(MODEL_PATH):
        print("错误：找不到模型文件")
        return
    model = ParamPredictorMLP(seq_len=WINDOW_SIZE, input_dim=9).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(SCALER_PATH, 'r') as f:
        scaler = json.load(f)
    mean = np.array(scaler['mean'], dtype=np.float32)
    std = np.array(scaler['std'], dtype=np.float32)

    # 3. 批量测试
    results = []
    for name, filepath in SCENARIOS.items():
        res = run_simulation(name, filepath, model, mean, std, device)
        if res:
            results.append(res)

    # 4. 生成汇总报告
    if not results:
        print("未产生有效测试结果。")
        return

    df_res = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("Multi-Scenario Generalization Test Report")
    print("=" * 100)
    cols = ["Scenario", "RMSE_Pos_Fixed", "RMSE_Pos_Adaptive", "Improv_Pos_%", "RMSE_Vel_Fixed", "RMSE_Vel_Adaptive",
            "Improv_Vel_%"]
    print(df_res[cols].to_string(index=False, float_format="%.4f"))
    print("-" * 100)

    avg_improv_pos = df_res["Improv_Pos_%"].mean()
    avg_improv_vel = df_res["Improv_Vel_%"].mean()
    print(f"Average Improvement (Position): {avg_improv_pos:.2f}%")
    print(f"Average Improvement (Velocity): {avg_improv_vel:.2f}%")

    # ================= 5. 绘制对比柱状图 (分开两张图) =================

    # --- 辅助函数：标注数值 ---
    def autolabel(ax, rects):
        """在柱子上方显示数值"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    x = np.arange(len(df_res))
    width = 0.35

    # ------------------ 图 1: 位置误差对比 (Position) ------------------
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()

    vals_fix_p = df_res["RMSE_Pos_Fixed"]
    vals_adp_p = df_res["RMSE_Pos_Adaptive"]

    rects1 = ax1.bar(x - width / 2, vals_fix_p, width, label='Fixed IMM', color='b', alpha=0.6)
    rects2 = ax1.bar(x + width / 2, vals_adp_p, width, label='NN-IMM', color='r', alpha=0.7)

    ax1.set_xlabel('Scenarios', fontsize=12)
    ax1.set_ylabel('RMSE Position (m)', fontsize=12)
    ax1.set_title(f'Position Error Comparison (Avg Imp: {avg_improv_pos:.1f}%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_res["Scenario"], rotation=15)
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 动态缩放 Y 轴 (位置)
    all_vals_p = np.concatenate([vals_fix_p, vals_adp_p])
    min_p, max_p = np.min(all_vals_p), np.max(all_vals_p)
    margin_p = (max_p - min_p) * 0.5 if max_p != min_p else 0.1
    # 顶部多留一点空间给数字标签
    ax1.set_ylim(max(0, min_p - margin_p), max_p + margin_p * 1.3)

    # 添加数值标签
    autolabel(ax1, rects1)
    autolabel(ax1, rects2)

    plt.tight_layout()
    # 如果想一次性看所有图，最后再 show，或者每画完一张就 show
    # 这里建议最后一起 show，或者在 IDE 里会自动弹窗

    # ------------------ 图 2: 速度误差对比 (Velocity) ------------------
    plt.figure(figsize=(10, 6))
    ax2 = plt.gca()

    vals_fix_v = df_res["RMSE_Vel_Fixed"]
    vals_adp_v = df_res["RMSE_Vel_Adaptive"]

    rects3 = ax2.bar(x - width / 2, vals_fix_v, width, label='Fixed IMM', color='b', alpha=0.6)
    rects4 = ax2.bar(x + width / 2, vals_adp_v, width, label='NN-IMM', color='r', alpha=0.7)

    ax2.set_xlabel('Scenarios', fontsize=12)
    ax2.set_ylabel('RMSE Velocity (m/s)', fontsize=12)
    ax2.set_title(f'Velocity Error Comparison (Avg Imp: {avg_improv_vel:.1f}%)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_res["Scenario"], rotation=15)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 动态缩放 Y 轴 (速度)
    all_vals_v = np.concatenate([vals_fix_v, vals_adp_v])
    min_v, max_v = np.min(all_vals_v), np.max(all_vals_v)
    margin_v = (max_v - min_v) * 0.5 if max_v != min_v else 0.1
    ax2.set_ylim(max(0, min_v - margin_v), max_v + margin_v * 1.3)

    # 添加数值标签
    autolabel(ax2, rects3)
    autolabel(ax2, rects4)

    plt.tight_layout()

    # 最后展示所有图表
    plt.show()


if __name__ == '__main__':
    main_multi_scenario()