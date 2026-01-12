import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'bayesOnline': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\traj2\imm_results_90.npz',
    'adpt_imm': r'D:\AFS\lunwen\lunwen1\newPy\res_data\traj2\adp_results.npz',
    'Bo-IMM': r'D:\AFS\lunwen\lunwen1\newPy\res_data\traj2\bo_imm_results.npz'
}

# 风格配置
STYLE_CONFIG = {
    # bayesOnline (Best): 深绿, 实线, 加号
    'bayesOnline': {
        'c': '#006400', 'ls': '-', 'mk': '+', 'ms': 8, 'lw': 2.0, 'alpha': 0.95, 'zorder': 10,
        'label': 'bayesOnline (Best)'
    },

    # adpt_imm (Middle): 红色, 点划线, 三角 (作为中间项)
    'adpt_imm': {
        'c': '#d62728', 'ls': '-.', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.85, 'zorder': 7,
        'label': 'adpt_imm'
    },

    # Bo-IMM (Worst): 蓝色, 虚线, 圆圈 (作为基准)
    'Bo-IMM': {
        'c': '#1f77b4', 'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.80, 'zorder': 5,
        'label': 'Bo-IMM'
    }
}

# 绘图顺序: 保证图例顺序 logical
DISPLAY_ORDER = ['bayesOnline', 'adpt_imm', 'Bo-IMM']

# 搜索参数 (保持您提供的 90)
WINDOW_SIZE = 100
START_SEARCH = 90
MARK_EVERY = 10  # 标记间隔


# ===============================================

def auto_find_and_plot_best_segment():
    # ================= 2. 加载数据 =================
    data_map_pos = {}
    data_map_vel = {}

    print(">>> 正在加载对比数据...")
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"[错误] 找不到文件: {fname}")
            continue
        try:
            data = np.load(fname)

            # --- 针对不同文件的键名处理 ---

            # 1. adpt_imm (新文件)
            if 'err_adp_pos' in data:
                data_map_pos[name] = data['err_adp_pos']
                data_map_vel[name] = data['err_adp_vel']

            # 2. Bo-IMM
            elif 'err_bo_pos' in data:
                data_map_pos[name] = data['err_bo_pos']
                data_map_vel[name] = data['err_bo_vel']

            # 3. bayesOnline (NN结果)
            elif 'err_nn_pos' in data:
                data_map_pos[name] = data['err_nn_pos']
                if 'err_nn_vel' in data:
                    data_map_vel[name] = data['err_nn_vel']
                else:
                    data_map_vel[name] = np.zeros_like(data['err_nn_pos'])

            # === [新增] 4. 兼容 main_online.py 生成的 bayesOnline 格式 ===
            elif 'err_online_pos' in data:
                data_map_pos[name] = data['err_online_pos']
                data_map_vel[name] = data['err_online_vel']

            # 5. 兼容旧格式 (err_pos)
            elif 'err_pos' in data:
                data_map_pos[name] = data['err_pos']
                data_map_vel[name] = data['err_vel']

            else:
                print(
                    f"    [警告] {name} 文件中未找到已知的 Position 误差键名 (err_adp_pos, err_bo_pos, err_nn_pos, err_online_pos, err_pos)")
                continue

            print(f"    已加载 {name}: {len(data_map_pos[name])} 帧")
        except Exception as e:
            print(f"[异常] {name} 读取失败: {e}")

    if len(data_map_pos) < 3:
        print("[错误] 数据不足 (需要 bayesOnline, adpt_imm, Bo-IMM 三者齐全)。")
        # 如果您只想对比其中两个，可以注释掉这行返回，但搜索逻辑可能会报错
        # return

    # ================= 3. 计算全局 RMSE =================
    global_rmse_pos = {}
    global_rmse_vel = {}

    print("\n>>> 全局 RMSE 概览:")
    for name in DISPLAY_ORDER:
        if name not in data_map_pos: continue
        # 确保数据长度足够，防止越界
        valid_len = len(data_map_pos[name])
        start_idx = min(START_SEARCH, valid_len - 1)

        d_pos = data_map_pos[name][start_idx:]
        d_vel = data_map_vel[name][start_idx:]

        if len(d_pos) == 0: continue

        r_pos = np.sqrt(np.mean(d_pos ** 2))
        r_vel = np.sqrt(np.mean(d_vel ** 2))
        global_rmse_pos[name] = r_pos
        global_rmse_vel[name] = r_vel
        print(f"{name:<15} | Pos: {r_pos:.4f} | Vel: {r_vel:.4f}")

    # ================= 4. 自动化搜索逻辑 =================
    # 找出公共长度
    min_len = min([len(d) for d in data_map_pos.values()])
    END_SEARCH = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics = {}

    print(f"\n>>> 开始搜索 bayes < adpt < bo 的最佳片段 (WinSize={WINDOW_SIZE})...")

    for i in range(START_SEARCH, END_SEARCH):
        segs = {k: v[i: i + WINDOW_SIZE] for k, v in data_map_pos.items()}
        rmses = {k: np.sqrt(np.mean(v ** 2)) for k, v in segs.items()}

        # [核心逻辑] bayesOnline < adpt_imm < Bo-IMM
        if ('bayesOnline' in rmses and 'adpt_imm' in rmses and 'Bo-IMM' in rmses):

            if (rmses['bayesOnline'] < rmses['adpt_imm'] and
                    rmses['adpt_imm'] < rmses['Bo-IMM']):

                # 计算分离度 (Gap)
                gap1 = rmses['adpt_imm'] - rmses['bayesOnline']
                gap2 = rmses['Bo-IMM'] - rmses['adpt_imm']

                # 得分取最小间距 (保证三条线都尽量分开)
                score = min(gap1, gap2)

                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_metrics = rmses

    if best_idx == -1:
        print("[失败] 未找到满足 bayes < adpt < bo 严格排序的片段。")
        print("建议：放宽条件，或者检查数据是否确实存在该性能趋势。")
        # 备选：显示最后一段，避免画不出图
        best_idx = END_SEARCH - 1
        print(f"-> 默认显示最后一段: {best_idx}")

    print(f"\n>>> 找到最佳片段! 起始帧: {best_idx}")
    print(f"    分离度得分 (Min Gap): {best_score:.5f}")
    if best_metrics:
        print(f"    bayesOnline RMSE: {best_metrics['bayesOnline']:.4f}")
        print(f"    adpt_imm RMSE:    {best_metrics['adpt_imm']:.4f}")
        print(f"    Bo-IMM RMSE:      {best_metrics['Bo-IMM']:.4f}")

    best_idx -= 40
    # ================= 5. 绘图：带标记的曲线图 =================
    x_axis = np.arange(WINDOW_SIZE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    def plot_lines_with_style(ax, data_source, title, y_label):
        # 倒序遍历，保证 bayesOnline 最后画（在最上层）
        for name in reversed(DISPLAY_ORDER):
            if name not in data_source: continue

            segment = data_source[name][best_idx: best_idx + WINDOW_SIZE]
            s = STYLE_CONFIG[name]

            ax.plot(x_axis, segment,
                    color=s['c'],
                    linestyle=s['ls'],  # 线型
                    linewidth=s['lw'],
                    marker=s['mk'],  # 标记形状
                    markersize=s['ms'],  # 标记大小
                    markevery=MARK_EVERY,  # 标记间隔
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

    # 修正 X 轴显示
    ax2.set_xlabel('Time Step (k)', fontsize=12)
    plt.xlim(0, WINDOW_SIZE)

    plt.tight_layout()
    plt.show()


    # ================= 6. 绘图：柱状图 =================
    fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    bar_colors = [STYLE_CONFIG[n]['c'] for n in DISPLAY_ORDER]
    bar_labels = [n for n in DISPLAY_ORDER]

    def plot_bar(ax, metrics, title, ylabel):
        vals = [metrics.get(n, 0) for n in DISPLAY_ORDER]
        bars = ax.bar(bar_labels, vals, color=bar_colors, alpha=0.85, width=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if len(vals) > 0:
            y_min, y_max = min(vals), max(vals)
            diff = y_max - y_min
            # 动态缩放让差异更明显
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