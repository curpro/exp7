import numpy as np
import matplotlib.pyplot as plt
import os

def auto_find_and_plot_best_segment():
    # ================= 1. 加载数据 =================
    # 定义文件名映射 (根据你的要求修改)
    files = {
        # 注意：这里保留了原代码中 Att 的路径，如果 Att 的路径也有变化，请手动修改这里
        'CNN-LSTM-Att': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
        'RNN': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\\5_RNN\nn_results_RNN.npz',
        'MLP': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\\4_MLP\nn_results_MLP.npz'
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

            # 读取速度误差 'err_nn_vel'
            if 'err_nn_vel' in loaded:
                data_map_vel[name] = loaded['err_nn_vel']
            else:
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
    print(f"{'Model':<15} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15}")
    print("-" * 50)

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

        print(f"{name:<15} | {rmse_pos:.4f}          | {rmse_vel:.4f}")

    # ================= 3. 自动化搜索逻辑 (基于位置误差) =================
    WINDOW_SIZE = 90 # 搜索窗口大小
    START_SEARCH = 90  # 跳过初始化
    # 使用 Att 的长度作为参考
    END_SEARCH = len(data_map_pos['CNN-LSTM-Att']) - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics_pos = {}

    print(f"\n>>> 开始全自动搜索 (范围: {START_SEARCH} - {END_SEARCH})...")
    print("    筛选条件(基于位置): RMSE(CNN-LSTM-Att) < RMSE(MLP) < RMSE(RNN)")

    # 遍历每一个可能的起始点
    for i in range(START_SEARCH, END_SEARCH):
        # 提取当前窗口数据
        segs = {name: data[i: i + WINDOW_SIZE] for name, data in data_map_pos.items()}

        # 1. 计算平均误差 (RMSE)
        rmses = {name: np.sqrt(np.mean(seg ** 2)) for name, seg in segs.items()}

        # 2. 计算“一致性” (防止曲线交叉)
        # 统计这100个点里，有多少个点严格满足 Att < RNN < MLP
        arr_att = segs['CNN-LSTM-Att']
        arr_rnn = segs['RNN']
        arr_mlp = segs['MLP']

        # 计算不交叉的点数 (严格满足 Att < RNN < MLP)
        valid_points = np.sum((arr_att < arr_mlp) & (arr_mlp < arr_rnn))
        consistency_rate = valid_points / WINDOW_SIZE

        # 3. 筛选条件：RMSE 顺序要对，且至少 60% 的点不交叉
        if (rmses['CNN-LSTM-Att'] < rmses['MLP'] < rmses['RNN']) and (consistency_rate > 0.6):

            # 4. 综合打分：结合“分离度”和“一致性”
            # 计算最大差距 (MLP - Att)
            gap = rmses['RNN'] - rmses['CNN-LSTM-Att']
            score = gap + (consistency_rate * 2.0)

            if score > best_score:
                best_score = score
                best_idx = i
                best_metrics_pos = rmses

    if best_idx == -1:
        print("[结果] 未找到严格满足该排序的片段。建议放宽条件或检查数据一致性。")
        # 如果找不到，为了防止报错，可以默认取最后一段或者直接返回
        # 这里选择直接返回，不做图
        return

    # 打印搜索结果
    print(f"\n>>> 找到最佳片段！起始帧: {best_idx}")
    print(f"    分类度得分 (Min Gap): {best_score:.4f}")
    print(f"    该片段位置 RMSE:")
    print(f"      CNN-LSTM-Att : {best_metrics_pos['CNN-LSTM-Att']:.4f}")
    print(f"      RNN          : {best_metrics_pos['RNN']:.4f}")
    print(f"      MLP          : {best_metrics_pos['MLP']:.4f}")

    # ================= 4. 绘图逻辑 (折线图) =================
    print("\n>>> 正在绘制对比图 (位置 & 速度)...")

    # 定义样式 (修改为3个模型)
    styles = {
        'CNN-LSTM-Att': {'c': '#006400', 'ls': '-', 'mk': 's', 'lw': 2.0, 'label': 'CNN-LSTM-Att'}, # 墨绿
        'RNN':          {'c': '#9467bd', 'ls': '--', 'mk': 'o', 'lw': 1.5, 'label': 'RNN'},          # 紫色
        'MLP':          {'c': '#d62728', 'ls': ':', 'mk': '^', 'lw': 1.5, 'label': 'MLP'}            # 红色
    }

    # 绘图顺序：误差大的先画，误差小的最后画（防止被遮挡）
    plot_order = ['RNN', 'MLP', 'CNN-LSTM-Att']
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
    # 使用 MLP (通常误差最大) 来确定 Y 轴上限
    max_val_pos = np.max(data_map_pos['RNN'][best_idx: best_idx + WINDOW_SIZE])
    ax1.set_ylim(0, max_val_pos * 1.25)

    # --- 子图 2: 速度误差 ---
    for name in plot_order:
        segment = data_map_vel[name][best_idx: best_idx + WINDOW_SIZE]
        s = styles[name]
        ax2.plot(x_axis, segment, color=s['c'], linestyle=s['ls'], linewidth=s['lw'],
                 marker=s['mk'], markersize=7, markevery=8, label=s['label'], alpha=0.9)

    ax2.set_title(f'Velocity Estimation Error Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Velocity Error (m/s)', fontsize=12)
    ax2.set_xlabel('Time Step (k)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 动态调整 Y 轴 (Vel)
    max_val_vel = np.max(data_map_vel['RNN'][best_idx: best_idx + WINDOW_SIZE])
    ax2.set_ylim(0, max_val_vel * 1.25)

    plt.xlim(0, WINDOW_SIZE - 1)
    plt.xticks(np.arange(0, WINDOW_SIZE + 1, 20))
    plt.tight_layout()
    plt.show()

    # ================= 5. 绘图逻辑 (柱状图) =================
    # 创建 1x2 的子图
    fig_bar, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # 柱状图顺序：Att -> RNN -> MLP
    bar_order = ['CNN-LSTM-Att', 'MLP', 'RNN']
    bar_labels = [styles[m]['label'] for m in bar_order]
    bar_colors = [styles[m]['c'] for m in bar_order]

    # 辅助函数
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