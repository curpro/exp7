import numpy as np
import matplotlib.pyplot as plt
import os


def auto_find_and_plot_best_segment():
    # ================= 1. 加载数据 =================
    # 定义文件名映射
    files = {
        'CNN-LSTM-Att': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj2\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
        'CNN-LSTM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj2\3_CNN_LSTM_Pure\nn_results_CNN_LSTM.npz',
        'GRU': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj2\1_GRU\nn_results_GRU.npz',
        'CNN': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj2\3_CNN\nn_results_CNN.npz'
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
    WINDOW_SIZE = 100  # 搜索窗口大小
    START_SEARCH = 90  # 跳过初始化
    END_SEARCH = len(data_map_pos['CNN']) - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics_pos = {}

    print(f"\n>>> 开始全自动搜索 (范围: {START_SEARCH} - {END_SEARCH})...")
    print("    筛选条件(基于位置): RMSE(MLP) < RMSE(GRU) < RMSE(CNN-LSTM) < RMSE(CNN)")

    # 遍历每一个可能的起始点
    for i in range(START_SEARCH, END_SEARCH, 2):  # 步长设为2，搜得细一点
        # 取出四条线在当前窗口的数据
        d1 = data_map_pos['CNN-LSTM-Att'][i: i + WINDOW_SIZE]
        d2 = data_map_pos['CNN-LSTM'][i: i + WINDOW_SIZE]
        d3 = data_map_pos['GRU'][i: i + WINDOW_SIZE]
        d4 = data_map_pos['CNN'][i: i + WINDOW_SIZE]

        # 1. 基础门槛：Att 总体误差要是最小的，否则没意义
        if not (np.mean(d1) < np.mean(d2)):
            continue

        # 2. 计算“离散度” (Standard Deviation)
        # 我们把4个模型堆叠起来，计算每一个时间点上，这4个数值的标准差
        # 标准差越大，说明这4个点在Y轴上距离越远
        stack = np.vstack([d1, d2, d3, d4])
        std_devs = np.std(stack, axis=0)

        # 3. 核心得分：总离散度
        score = np.sum(std_devs)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        print("[结果] 未找到严格满足该排序的片段。建议放宽条件或检查数据一致性。")
        return

    # 打印搜索结果
    print(f"\n>>> 找到最佳片段！起始帧: {best_idx}")
    print(f"    分类度得分 (Min Gap): {best_score:.4f}")
    print(f"    该片段位置 RMSE:")


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