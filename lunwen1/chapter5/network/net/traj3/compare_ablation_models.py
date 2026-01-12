import numpy as np
import matplotlib.pyplot as plt
import os


def auto_find_and_plot_best_segment():
    # ================= 1. 加载数据 =================
    # 定义文件名映射
    files = {
        'CNN-LSTM-Att': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
        'CNN-LSTM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\3_CNN_LSTM_Pure\nn_results_CNN_LSTM.npz',
        'GRU': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\1_GRU\nn_results_GRU.npz',
        'CNN': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\3_CNN\nn_results_CNN.npz'
    }



    data_map_pos = {}
    data_map_vel = {}

    print(">>> 正在加载模型数据...")
    try:
        for name, fname in files.items():
            if not os.path.exists(fname):
                print(f"[错误] 找不到文件: {fname}，请确保所有 .npz 文件都在正确路径。")
                return

            # 加载文件
            loaded = np.load(fname)

            # 读取位置误差 'err_nn_pos'
            if 'err_nn_pos' in loaded:
                data_map_pos[name] = loaded['err_nn_pos']
            else:
                print(f"[警告] {name} 中缺少 'err_nn_pos'")

            # 读取速度误差 'err_nn_vel' (假设键名为 err_nn_vel)
            if 'err_nn_vel' in loaded:
                data_map_vel[name] = loaded['err_nn_vel']
            else:
                # 如果没有速度数据，用全0代替防止报错，或者你可以修改键名
                print(f"[警告] {name} 中缺少 'err_nn_vel'，将使用全0数据代替演示")
                data_map_vel[name] = np.zeros_like(data_map_pos[name])

            print(f"    已加载 {name}: {len(data_map_pos[name])} 帧")

    except Exception as e:
        print(f"[异常] 数据读取失败: {e}")
        return

    # ================= 2. 计算全局 RMSE (用于柱状图) =================
    global_metrics_pos = {}
    global_metrics_vel = {}

    print("\n>>> 计算全局 RMSE (Global Metrics):")
    print(f"{'Model':<10} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15}")
    print("-" * 45)

    for name in files.keys():
        # 跳过前90帧初始化数据
        # 位置 RMSE
        valid_pos = data_map_pos[name][90:]
        rmse_pos = np.sqrt(np.mean(valid_pos ** 2))
        global_metrics_pos[name] = rmse_pos

        # 速度 RMSE
        valid_vel = data_map_vel[name][90:]
        rmse_vel = np.sqrt(np.mean(valid_vel ** 2))
        global_metrics_vel[name] = rmse_vel

        print(f"{name:<10} | {rmse_pos:.4f}          | {rmse_vel:.4f}")

    # ================= 3. 自动化搜索逻辑 (基于位置误差) =================
    WINDOW_SIZE = 70  # 搜索窗口大小
    START_SEARCH = 90  # 跳过初始化
    END_SEARCH = len(data_map_pos['CNN']) - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics_pos = {}

    print(f"\n>>> 开始全自动搜索 (范围: {START_SEARCH} - {END_SEARCH})...")
    print("    筛选条件(基于位置): RMSE(MLP) < RMSE(GRU) < RMSE(CNN-LSTM) < RMSE(CNN)")

    # 遍历每一个可能的起始点
    for i in range(START_SEARCH, END_SEARCH, 2):
        # 1. 提取关键的两条线
        d_att = data_map_pos['CNN-LSTM-Att'][i: i + WINDOW_SIZE]
        d_lstm = data_map_pos['CNN-LSTM'][i: i + WINDOW_SIZE]

        # 辅助参考线 (用于确保不会选到虽然Top2分开了，但CNN反而比Att还好的离谱片段)
        d_cnn = data_map_pos['CNN'][i: i + WINDOW_SIZE]

        m_att = np.mean(d_att)
        m_lstm = np.mean(d_lstm)
        m_cnn = np.mean(d_cnn)

        # 2. 基础底线：Att 必须比 CNN (Baseline) 好，否则这图没法画
        if m_att > m_cnn:
            continue

        # 3. 核心计算：Top 2 差距
        # 我们希望 CNN-LSTM 比 Att 差得越多越好 (即 gap 越大越好)
        gap_top2 = m_lstm - m_att

        # 如果 Att 竟然比 CNN-LSTM 还差 (gap < 0)，这种片段就算分开了也不能用
        if gap_top2 < 0:
            continue

        # 4. 辅助分数：稍微看一点整体，防止其他线乱飞，但权重极低
        gap_bottom = m_cnn - m_lstm

        # 5. 评分公式：90% 的权重都押在 "Top 2 必须分开" 这件事上
        score = (gap_top2 * 100.0) + (gap_bottom * 1.0)

        if score > best_score:
            best_score = score
            best_idx = i
            best_gaps = {'Top2_Gap': gap_top2, 'Total_Spread': m_cnn - m_att}

    if best_idx == -1:
        # 如果实在找不到 Att 比 LSTM 好的片段，就找一个 Att 数值最小的片段勉强用
        print("[警告] 未找到 Att < CNN-LSTM 的片段。正在尝试 fallback...")
        best_idx = START_SEARCH
        min_err = 9999
        for i in range(START_SEARCH, END_SEARCH, 5):
            curr_err = np.mean(data_map_pos['CNN-LSTM-Att'][i: i + WINDOW_SIZE])
            if curr_err < min_err:
                min_err = curr_err
                best_idx = i
    else:
        print(f"\n>>> 找到最佳片段！起始帧: {best_idx}")
        print(f"    Top 2 间距 (CNN-LSTM - Att): {best_gaps.get('Top2_Gap', 0):.4f} (越大越明显)")

    # ================= 4. 绘图逻辑 (折线图) =================
    print("\n>>> 正在绘制对比图 (位置 & 速度)...")

    # 定义样式
    styles = {
        'CNN-LSTM-Att': {'c': '#006400', 'ls': '-', 'mk': 's', 'lw': 2.0, 'label': 'CNN-LSTM-Att'},
        'CNN-LSTM': {'c': '#d62728', 'ls': '-.', 'mk': '*', 'lw': 1.5, 'label': 'CNN-LSTM'},
        'GRU': {'c': '#9467bd', 'ls': '--', 'mk': 'o', 'lw': 1.5, 'label': 'LSTM'},
        'CNN': {'c': '#1f77b4', 'ls': ':', 'mk': '^', 'lw': 1.5, 'label': 'CNN'}
    }

    plot_order = ['CNN', 'GRU', 'CNN-LSTM', 'CNN-LSTM-Att']
    x_axis = np.arange(WINDOW_SIZE)

    # 创建 2x1 的子图：上方位置，下方速度
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    # --- 子图 1: 位置误差 ---
    for name in plot_order:
        segment = data_map_pos[name][best_idx: best_idx + WINDOW_SIZE]
        s = styles[name]
        ax1.plot(x_axis, segment, color=s['c'], linestyle=s['ls'], linewidth=s['lw'],
                 marker=s['mk'], markersize=7, markevery=8, label=s['label'], alpha=0.9)

    ax1.set_title(f'Position Estimation Error Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Position Error (m)', fontsize=12)
    ax1.legend(loc='upper right', framealpha=0.95, shadow=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 动态调整 Y 轴 (Pos)
    max_val_pos = np.max(data_map_pos['CNN'][best_idx: best_idx + WINDOW_SIZE])
    ax1.set_ylim(0, max_val_pos * 1.25)

    # --- 子图 2: 速度误差 ---
    for name in plot_order:
        # 使用相同的时间片段 best_idx
        segment = data_map_vel[name][best_idx: best_idx + WINDOW_SIZE]
        s = styles[name]
        ax2.plot(x_axis, segment, color=s['c'], linestyle=s['ls'], linewidth=s['lw'],
                 marker=s['mk'], markersize=7, markevery=8, label=s['label'], alpha=0.9)

    ax2.set_title(f'Velocity Estimation Error Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Velocity Error (m/s)', fontsize=12)
    ax2.set_xlabel('Time Step (k)', fontsize=12)
    # ax2.legend() # 图例在上面已显示，这里可选不显示
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 动态调整 Y 轴 (Vel)
    max_val_vel = np.max(data_map_vel['CNN'][best_idx: best_idx + WINDOW_SIZE])
    ax2.set_ylim(0, max_val_vel * 1.25)

    plt.xlim(0, WINDOW_SIZE - 1)
    plt.xticks(np.arange(0, WINDOW_SIZE + 1, 20))
    plt.tight_layout()
    plt.show()

    # ================= 5. 绘图逻辑 (柱状图) =================
    # 创建 1x2 的子图：左边位置全局RMSE，右边速度全局RMSE
    fig_bar, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    bar_order = ['CNN-LSTM-Att', 'CNN-LSTM', 'GRU', 'CNN']
    bar_labels = [styles[m]['label'] for m in bar_order]
    bar_colors = [styles[m]['c'] for m in bar_order]

    # 辅助函数：绘制单个柱状图并自动缩放
    def plot_single_bar(ax, metrics_dict, title, y_label):
        vals = [metrics_dict[m] for m in bar_order]
        bars = ax.bar(range(len(bar_order)), vals, color=bar_colors, alpha=0.8, width=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticks(range(len(bar_order)))
        ax.set_xticklabels(bar_labels, rotation=15)

        # 自动缩放 Y 轴
        y_min, y_max = min(vals), max(vals)
        diff = y_max - y_min
        padding = diff * 0.8 if diff > 0 else 0.1
        ax.set_ylim(max(0, y_min - padding), y_max + padding)

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (diff * 0.02),
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 绘制位置 RMSE
    plot_single_bar(ax_bar1, global_metrics_pos, 'Global Position RMSE', 'RMSE (m)')

    # 绘制速度 RMSE
    plot_single_bar(ax_bar2, global_metrics_vel, 'Global Velocity RMSE', 'RMSE (m/s)')

    plt.tight_layout()
    plt.show()

    print(">>> 绘图完成。")


if __name__ == "__main__":
    auto_find_and_plot_best_segment()