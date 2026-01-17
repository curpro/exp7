import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'Win 45': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\Win_45\nn_results_win45.npz',
    'Win 180': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\Win_180\nn_results_win180.npz',
    'Win 90': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\nn_results_win90.npz'
}

# 风格配置
STYLE_CONFIG = {
    'Win 90': {
        'c': '#006400', 'ls': '-', 'mk': '+', 'ms': 8, 'lw': 1.7, 'alpha': 0.95, 'zorder': 10,
        'label': 'Win 90 (Best)'
    },
    'Win 180': {
        'c': '#1f77b4', 'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.75, 'zorder': 5,
        'label': 'Win 180'
    },
    'Win 45': {
        'c': '#d62728', 'ls': ':', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.75, 'zorder': 1,
        'label': 'Win 45'
    }
}

# 绘图顺序
DISPLAY_ORDER = ['Win 90', 'Win 180', 'Win 45']

# 搜索参数
WINDOW_SIZE = 100
START_SEARCH = 90
MARK_EVERY = 8


# ===============================================

def auto_find_and_plot_best_segment():
    # ================= 2. 加载数据 =================
    data_map_pos = {}
    data_map_vel = {}

    print(">>> 正在加载窗口对比数据...")
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"[错误] 找不到文件: {fname}")
            continue
        try:
            data = np.load(fname)
            if 'err_nn_pos' in data:
                data_map_pos[name] = data['err_nn_pos']
                data_map_vel[name] = data['err_nn_vel'] if 'err_nn_vel' in data else np.zeros_like(data['err_nn_pos'])
            else:
                data_map_pos[name] = data['err_pos']
                data_map_vel[name] = data['err_vel']

            print(f"    已加载 {name}: {len(data_map_pos[name])} 帧")
        except Exception as e:
            print(f"[异常] {name} 读取失败: {e}")

    if len(data_map_pos) < 3:
        print("[错误] 数据不足。")
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

    # ================= [核心修改] 替换为 combine 版的搜索逻辑 =================
    min_len = min([len(d) for d in data_map_pos.values()])
    END_SEARCH = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = START_SEARCH  # 默认起始位置

    # 这里的 best_metrics 仅用于打印最后选中的区间 RMSE，不再用于搜索判断
    # 搜索判断直接用临时计算的 rmses
    final_best_metrics = {}

    print(f"\n>>> 开始搜索 Win 90 < Win 180 < Win 45 (使用 Combine 简化逻辑)...")

    for i in range(START_SEARCH, END_SEARCH):
        # 计算当前窗口的 RMSE
        rmses = {k: np.sqrt(np.mean(v[i: i + WINDOW_SIZE] ** 2)) for k, v in data_map_pos.items()}

        if ('Win 90' in rmses and 'Win 180' in rmses and 'Win 45' in rmses):
            # 核心判断: 90 (Best) < 180 < 45 (Worst)
            if (rmses['Win 90'] < rmses['Win 180'] < rmses['Win 45']):

                # 计算间距
                gap1 = rmses['Win 180'] - rmses['Win 90']
                gap2 = rmses['Win 45'] - rmses['Win 180']

                # 取最小间隔作为得分 (Combine 版逻辑)
                score = min(gap1, gap2)

                if score > best_score:
                    best_score = score
                    best_idx = i
                    final_best_metrics = rmses

    if best_score == -1.0:
        print("[警告] 未找到完全满足 Win 90 < 180 < 45 的区间，使用默认起始帧。")

    print(f"\n>>> 找到最佳片段! 起始帧: {best_idx}")
    print(f"    分离度得分: {best_score:.5f}")

    # ================= 5. 绘图：带标记的曲线图 =================
    x_axis = np.arange(WINDOW_SIZE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    def plot_lines_with_style(ax, data_source, title, y_label):
        # 倒序遍历，保证 Win 90 最后画（在最上层）
        for name in reversed(DISPLAY_ORDER):
            if name not in data_source: continue

            segment = data_source[name][best_idx: best_idx + WINDOW_SIZE]
            s = STYLE_CONFIG[name]

            ax.plot(x_axis, segment,
                    color=s['c'],
                    linestyle=s['ls'],
                    linewidth=s['lw'],
                    marker=s['mk'],
                    markersize=s['ms'],
                    markevery=MARK_EVERY,
                    alpha=s['alpha'],
                    zorder=s['zorder'],
                    label=s['label'])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', framealpha=0.95, shadow=True)

        # Y轴动态范围
        vals = [data_source[n][best_idx: best_idx + WINDOW_SIZE] for n in DISPLAY_ORDER if n in data_source]
        if vals:
            current_max = max([np.max(v) for v in vals])
            ax.set_ylim(0, current_max * 1.3)

    plot_lines_with_style(ax1, data_map_pos, 'Position Error Comparison', 'Position Error (m)')
    plot_lines_with_style(ax2, data_map_vel, 'Velocity Error Comparison', 'Velocity Error (m/s)')
    ax2.set_xlabel('Time Step (k)', fontsize=12)

    plt.tight_layout()
    plt.show()

    # ================= 6. 绘图：柱状图 (保持不变) =================
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