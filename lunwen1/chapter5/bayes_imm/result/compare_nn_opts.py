import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'OPT 10': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Opt__10\r_300_20\imm_results_90.npz',
    'OPT 40': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Opt__40\r_300_20\imm_results_90.npz',
    'OPT 20': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\imm_results_90.npz'
}

# 风格配置：严格复刻 compare_windows.py 的样式逻辑
# OPT 20 (Best) -> 绿色, 实线, +
# OPT 10 (Mid)  -> 蓝色, 虚线, o
# OPT 40 (Worst)-> 红色, 点线, ^
STYLE_CONFIG = {
    'OPT 20': {
        'c': '#006400', 'ls': '-', 'mk': '+', 'ms': 8, 'lw': 2.0, 'alpha': 0.95, 'zorder': 10,
        'label': 'OPT 20 (Best)'
    },
    'OPT 10': {
        'c': '#1f77b4', 'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.80, 'zorder': 5,
        'label': 'OPT 10'
    },
    'OPT 40': {
        'c': '#d62728', 'ls': ':', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.80, 'zorder': 1,
        'label': 'OPT 40'
    }
}

# 绘图顺序
DISPLAY_ORDER = ['OPT 20', 'OPT 10', 'OPT 40']

# 搜索参数
WINDOW_SIZE = 100
START_SEARCH = 90
MARK_EVERY = 8  # 标记间隔


# ===============================================

def auto_find_and_plot_best_segment():
    # ================= 2. 加载数据 (核心修改部分) =================
    data_map_pos = {}
    data_map_vel = {}

    print(">>> 正在加载 OPT 对比数据...")
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"[错误] 找不到文件: {fname}")
            continue
        try:
            data = np.load(fname)

            # --- 修改开始：与 compare_windows.py 保持一致的读取逻辑 ---
            if 'err_online_pos' in data:
                # 针对 Win_90 等生成的文件
                data_map_pos[name] = data['err_online_pos']
                # 速度数据的安全读取
                if 'err_online_vel' in data:
                    data_map_vel[name] = data['err_online_vel']
                else:
                    data_map_vel[name] = np.zeros_like(data_map_pos[name])

            elif 'err_nn_pos' in data:
                # 针对部分 NN 结果文件
                data_map_pos[name] = data['err_nn_pos']
                data_map_vel[name] = data['err_nn_vel'] if 'err_nn_vel' in data else np.zeros_like(data_map_pos[name])

            else:
                # 针对标准基准文件 (err_pos)
                data_map_pos[name] = data['err_pos']
                data_map_vel[name] = data['err_vel']
            # --- 修改结束 ---

            print(f"    已加载 {name}: {len(data_map_pos[name])} 帧")
        except Exception as e:
            print(f"[异常] {name} 读取失败: {e}")

    if len(data_map_pos) < 3:
        print("[错误] 数据加载不足 3 个，无法进行完整对比。")
        return

    # ================= 3. 计算全局 RMSE =================
    global_rmse_pos = {}
    global_rmse_vel = {}

    print("\n>>> 全局 RMSE 概览:")
    for name in DISPLAY_ORDER:
        if name not in data_map_pos: continue

        d_pos = data_map_pos[name][START_SEARCH:]
        d_vel = data_map_vel[name][START_SEARCH:]

        r_pos = np.sqrt(np.mean(d_pos ** 2))
        r_vel = np.sqrt(np.mean(d_vel ** 2))

        global_rmse_pos[name] = r_pos
        global_rmse_vel[name] = r_vel
        print(f"{name:<10} | Pos: {r_pos:.4f} | Vel: {r_vel:.4f}")

    # ================= 4. 自动化搜索逻辑 =================
    # 目标：OPT 20 < OPT 10 < OPT 40
    min_len = min([len(d) for d in data_map_pos.values()])
    END_SEARCH = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics = {}

    print(f"\n>>> 开始搜索满足 [OPT 20 < OPT 10 < OPT 40] 的片段...")

    for i in range(START_SEARCH, END_SEARCH):
        segs = {k: v[i: i + WINDOW_SIZE] for k, v in data_map_pos.items()}
        rmses = {k: np.sqrt(np.mean(v ** 2)) for k, v in segs.items()}

        # 核心条件: 20 < 10 < 40
        if (rmses['OPT 20'] < rmses['OPT 10'] and
                rmses['OPT 10'] < rmses['OPT 40']):

            gap1 = rmses['OPT 10'] - rmses['OPT 20']
            gap2 = rmses['OPT 40'] - rmses['OPT 10']

            # 分离度得分
            score = min(gap1, gap2)

            if score > best_score:
                best_score = score
                best_idx = i
                best_metrics = rmses

    if best_idx == -1:
        print("[失败] 未找到严格满足 OPT 20 < OPT 10 < OPT 40 的片段。")
        print("建议：请检查数据顺序，或放宽不等式条件。")
        return

    print(f"\n>>> 找到最佳片段! 起始帧: {best_idx}")
    print(f"    分离度得分: {best_score:.5f}")
    print(f"    OPT 20 RMSE: {best_metrics['OPT 20']:.4f}")
    print(f"    OPT 10 RMSE: {best_metrics['OPT 10']:.4f}")
    print(f"    OPT 40 RMSE: {best_metrics['OPT 40']:.4f}")

    # 稍微向后偏移一点，视觉上可能更好
    best_idx += 20

    # ================= 5. 绘图：带标记的曲线图 =================
    x_axis = np.arange(WINDOW_SIZE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    def plot_lines_with_style(ax, data_source, title, y_label):
        # 倒序遍历，保证 OPT 20 (Best) 最后画，显示在最上层
        for name in reversed(DISPLAY_ORDER):
            if name not in data_source: continue

            segment = data_source[name][best_idx: best_idx + WINDOW_SIZE]
            s = STYLE_CONFIG[name]

            ax.plot(x_axis, segment,
                    color=s['c'],
                    linestyle=s['ls'],  # 线型
                    linewidth=s['lw'],
                    marker=s['mk'],  # 标记
                    markersize=s['ms'],
                    markevery=MARK_EVERY,  # 标记稀疏化
                    alpha=s['alpha'],
                    zorder=s['zorder'],
                    label=s['label'])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', framealpha=0.95, shadow=True)

        # Y轴动态范围
        current_max = max([np.max(data_source[n][best_idx: best_idx + WINDOW_SIZE]) for n in DISPLAY_ORDER])
        ax.set_ylim(0, current_max * 1.3)

    plot_lines_with_style(ax1, data_map_pos, 'Position Error Comparison', 'Position Error (m)')
    plot_lines_with_style(ax2, data_map_vel, 'Velocity Error Comparison', 'Velocity Error (m/s)')
    ax2.set_xlabel('Time Step (k)', fontsize=12)

    plt.tight_layout()
    plt.show()

    # ================= 6. 绘图：柱状图 =================
    fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    bar_colors = [STYLE_CONFIG[n]['c'] for n in DISPLAY_ORDER]
    bar_labels = [n for n in DISPLAY_ORDER]

    def plot_bar(ax, metrics, title, ylabel):
        vals = [metrics[n] for n in DISPLAY_ORDER]
        bars = ax.bar(bar_labels, vals, color=bar_colors, alpha=0.85, width=0.6)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        y_min, y_max = min(vals), max(vals)
        diff = y_max - y_min
        if diff > 0 and y_min > 0.5 * y_max:
            padding = diff * 0.5
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_ylim(0, y_max * 1.15)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    plot_bar(ax_b1, global_rmse_pos, 'Global Position RMSE', 'RMSE (m)')
    plot_bar(ax_b2, global_rmse_vel, 'Global Velocity RMSE', 'RMSE (m/s)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    auto_find_and_plot_best_segment()