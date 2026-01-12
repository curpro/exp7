import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置路径与样式 (与 compare_models.py 保持一致) =================
# 请确保路径与您的实际环境一致
FILES = {
    'CNN-LSTM-Att': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj1\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
    'CNN-LSTM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj1\3_CNN_LSTM_Pure\nn_results_CNN_LSTM.npz',
    'GRU': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj1\1_GRU\nn_results_GRU.npz',
    'CNN': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj1\3_CNN\nn_results_CNN.npz'
}

# 统一样式定义
STYLES = {
    'CNN-LSTM-Att': {'c': '#006400', 'ls': '-', 'lw': 1.5, 'label': 'CNN-LSTM-Att'},
    'CNN-LSTM': {'c': '#d62728', 'ls': '-', 'lw': 1.0, 'label': 'CNN-LSTM'},
    'GRU': {'c': '#9467bd', 'ls': '-', 'lw': 1.0, 'label': 'LSTM'},
    'CNN': {'c': '#1f77b4', 'ls': '-', 'lw': 1.0, 'label': 'CNN'}
}

# 绘图顺序
PLOT_ORDER = ['CNN', 'GRU', 'CNN-LSTM', 'CNN-LSTM-Att']

DT = 1 / 30
UNIFIED_START_FRAME = 90  # 忽略前 90 帧初始化误差


# ======================================

def load_and_compare_global():
    data_map_pos = {}
    data_map_vel = {}
    time_axis = None

    print(f"=== 开始读取全局数据 (跳过前 {UNIFIED_START_FRAME} 帧) ===")

    # 1. 加载数据
    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}")
            continue

        try:
            data = np.load(filename)

            # 读取位置和速度误差
            if 'err_nn_pos' in data:
                data_map_pos[label] = data['err_nn_pos']

            # 兼容性处理：如果没找到 err_nn_vel，用全0填充
            if 'err_nn_vel' in data:
                data_map_vel[label] = data['err_nn_vel']
            else:
                print(f"[提示] {label} 缺少速度数据，使用全0填充。")
                data_map_vel[label] = np.zeros_like(data['err_nn_pos'])

            # 统一时间轴 (只取一次)
            if time_axis is None and 't' in data:
                time_axis = data['t']

            print(f"    已加载: {label}")

        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")

    if not data_map_pos:
        print("错误：未加载到任何数据。")
        return

    # 如果没有时间轴，创建一个
    first_key = list(data_map_pos.keys())[0]
    total_len = len(data_map_pos[first_key])
    if time_axis is None:
        time_axis = np.arange(total_len) * DT

    # 2. 计算全局指标
    global_metrics_pos = {}
    global_metrics_vel = {}

    print("\n" + "=" * 60)
    print(f"{'Model':<15} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15}")
    print("-" * 60)

    for key in PLOT_ORDER:
        if key not in data_map_pos: continue

        # 截取有效片段
        err_pos = data_map_pos[key][UNIFIED_START_FRAME:]
        err_vel = data_map_vel[key][UNIFIED_START_FRAME:]

        rmse_pos = np.sqrt(np.mean(err_pos ** 2))
        rmse_vel = np.sqrt(np.mean(err_vel ** 2))

        global_metrics_pos[key] = rmse_pos
        global_metrics_vel[key] = rmse_vel

        print(f"{STYLES[key]['label']:<15} | {rmse_pos:<15.4f} | {rmse_vel:<15.4f}")

    print("=" * 60 + "\n")

    # ================= 绘图 1: 全局 RMSE 柱状图 =================
    # 逻辑与 compare_models.py 的柱状图部分完全一致
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # 准备绘图数据 (倒序排列，让最好的模型显示在特定位置，或保持 PLOT_ORDER)
    # 这里我们使用 PLOT_ORDER 的逆序，通常把最好的放在最左边或者最右边
    # 为了和 compare_models.py 一致，我们使用 bar_order
    bar_order = ['CNN-LSTM-Att', 'CNN-LSTM', 'GRU', 'CNN']
    bar_labels = [STYLES[m]['label'] for m in bar_order]
    bar_colors = [STYLES[m]['c'] for m in bar_order]

    def plot_single_bar(ax, metrics_dict, title, y_label):
        vals = [metrics_dict[m] for m in bar_order]
        bars = ax.bar(range(len(bar_order)), vals, color=bar_colors, alpha=0.8, width=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticks(range(len(bar_order)))
        ax.set_xticklabels(bar_labels, rotation=15)

        # 自动缩放 Y 轴 (保留视觉差异)
        y_min, y_max = min(vals), max(vals)
        diff = y_max - y_min
        # 如果差异太小，给一点默认边距
        padding = diff * 0.8 if diff > 1e-5 else y_max * 0.2
        ax.set_ylim(max(0, y_min - padding), y_max + padding)

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (diff * 0.05),
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plot_single_bar(ax1, global_metrics_pos, 'Global Position RMSE Comparison', 'RMSE (m)')
    plot_single_bar(ax2, global_metrics_vel, 'Global Velocity RMSE Comparison', 'RMSE (m/s)')

    fig1.tight_layout()

    # ================= 绘图 2: 全局误差演变曲线 (Full Timeline) =================
    plt.figure(figsize=(14, 10))

    # 定义 X 轴 (裁掉前 90 帧)
    plot_time = time_axis[UNIFIED_START_FRAME:]

    # --- 上图：位置误差 ---
    ax_pos = plt.subplot(2, 1, 1)

    for key in PLOT_ORDER:
        if key not in data_map_pos: continue

        data_seq = data_map_pos[key][UNIFIED_START_FRAME:]
        s = STYLES[key]

        # 全局图数据量大，markevery 设大一点避免密密麻麻，或者干脆不画 marker
        # 这里设置为每 50 帧画一个 marker，保持风格一致性
        ax_pos.plot(plot_time, data_seq,
                    color=s['c'], linestyle=s['ls'], linewidth=s['lw'],  # 全局图线宽稍微细一点
                    markersize=4, markevery=100,
                    label=f"{s['label']} (RMSE={global_metrics_pos[key]:.3f})",
                    alpha=0.8)

    ax_pos.set_title('Global Position Error Evolution', fontsize=14, fontweight='bold')
    ax_pos.set_ylabel('Position Error (m)', fontsize=12)
    ax_pos.legend(loc='upper right', framealpha=0.9, shadow=True)
    ax_pos.grid(True, linestyle='--', alpha=0.4)
    # 自动缩放 Y 轴以排除极端的初始化尖峰（如果还有的话）
    # ax_pos.set_ylim(0, np.percentile(data_map_pos['CNN'][UNIFIED_START_FRAME:], 99.5) * 1.5)

    # --- 下图：速度误差 ---
    ax_vel = plt.subplot(2, 1, 2)

    for key in PLOT_ORDER:
        if key not in data_map_vel: continue

        data_seq = data_map_vel[key][UNIFIED_START_FRAME:]
        s = STYLES[key]

        ax_vel.plot(plot_time, data_seq,
                    color=s['c'], linestyle=s['ls'], linewidth=s['lw'],
                    markersize=4, markevery=100,
                    label=f"{s['label']} (RMSE={global_metrics_vel[key]:.3f})",
                    alpha=0.8)

    ax_vel.set_title('Global Velocity Error Evolution', fontsize=14, fontweight='bold')
    ax_vel.set_ylabel('Velocity Error (m/s)', fontsize=12)
    ax_vel.set_xlabel('Time (s)', fontsize=12)
    ax_vel.legend(loc='upper right', framealpha=0.9, shadow=True)
    ax_vel.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_compare_global()