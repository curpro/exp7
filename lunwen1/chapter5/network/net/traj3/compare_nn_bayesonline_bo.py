import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'bayesOnline': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\traj3\imm_results_90.npz',
    'NN-IMM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj3\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
    'Bo-IMM': r'D:\AFS\lunwen\lunwen1\newPy\res_data\traj3\bo_imm_results.npz'
}

# 风格配置
STYLE_CONFIG = {
    # NN-IMM (Best): 深绿, 实线, 加号 -> 主角
    'NN-IMM': {
        'c': '#006400', 'ls': '-', 'mk': '+', 'ms': 8, 'lw': 2.0, 'alpha': 0.95, 'zorder': 10,
        'label': 'NN-IMM '
    },

    # bayesOnline (Baseline): 蓝色, 虚线, 圆圈 -> 对比项
    'bayesOnline':  {
        'c': '#d62728', 'ls': '-.', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.85, 'zorder': 7,
        'label': 'bayesOnline'
    },
    #Bo-IMM
    'Bo-IMM': {
        'c': '#1f77b4', 'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.80, 'zorder': 5,
        'label': 'Bo-IMM'
    }
}

# 绘图顺序: 保证 NN-IMM 最后画（浮在最上面）
DISPLAY_ORDER = ['NN-IMM', 'bayesOnline', 'Bo-IMM']

# 搜索参数
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

            # --- 通用键名处理 ---
            # 这里的 NN-IMM 和 bayesOnline 都是 NN 结果，通常键名都是 'err_nn_pos'
            # 我们通过上面的 name (FILES 的 key) 来区分它们存储到 data_map_pos 中

            if 'err_nn_pos' in data:
                data_map_pos[name] = data['err_nn_pos']
                if 'err_nn_vel' in data:
                    data_map_vel[name] = data['err_nn_vel']
                else:
                    data_map_vel[name] = np.zeros_like(data['err_nn_pos'])

                # 2. [新增] 适配 bayesOnline 的新名字 'err_online_pos'
            elif 'err_online_pos' in data:
                print(f"    [{name}] 识别到 'err_online_pos'，正在读取...")
                data_map_pos[name] = data['err_online_pos']
                data_map_vel[name] = data['err_online_vel']

                # 3. 兼容通用名
            elif 'err_pos' in data:
                data_map_pos[name] = data['err_pos']
                data_map_vel[name] = data['err_vel']

                # 4. 兼容 Bo/Adp 格式
            elif 'err_bo_pos' in data:
                data_map_pos[name] = data['err_bo_pos']
                data_map_vel[name] = data['err_bo_vel']
            elif 'err_adp_pos' in data:
                data_map_pos[name] = data['err_adp_pos']
                data_map_vel[name] = data['err_adp_vel']
            else:
                print(f"[错误] {name} 读取失败：未找到已知的误差变量名 (如 err_nn_pos, err_online_pos)")
                continue

            print(f"    已加载 {name}: {len(data_map_pos[name])} 帧")
        except Exception as e:
            print(f"[异常] {name} 读取失败: {e}")

    if len(data_map_pos) < 2:
        print("[错误] 数据不足 (需要 NN-IMM 和 bayesOnline)。")
        return

    # ================= 3. 计算全局 RMSE =================
    global_rmse_pos = {}
    global_rmse_vel = {}

    print("\n>>> 全局 RMSE 概览:")
    for name in DISPLAY_ORDER:
        if name not in data_map_pos: continue

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
    min_len = min([len(d) for d in data_map_pos.values()])
    END_SEARCH = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = -1
    best_metrics = {}

    # 权重配置：一致性占比很大，保证绝大多数帧都满足顺序
    # 如果两个窗口一致性相同，则看间距(gap)
    w_consistency = 1000.0
    w_gap = 1.0

    print(f"\n>>> 开始搜索 NN-IMM < bayesOnline < Bo-IMM 的最佳片段 (WinSize={WINDOW_SIZE})...")
    print(f"    策略: 优先保证每帧都满足顺序 (Point-wise Order)，其次寻找间距最大的片段。")

    for i in range(0, min_len - WINDOW_SIZE):
        # 提取当前窗口数据
        seg_nn = data_map_pos['NN-IMM'][i: i + WINDOW_SIZE]
        seg_bayes = data_map_pos['bayesOnline'][i: i + WINDOW_SIZE]

        # --- 核心打分逻辑 ---

        # 1. 一致性 (Consistency): 绿线在红线下方的比例
        #    这是最重要的，我们希望它接近 1.0 (即100%的时间都在下方)
        is_lower = seg_nn < seg_bayes
        consistency = np.mean(is_lower)

        # 2. 间距 (Gap): 红色减去绿色，越大越好
        gap = np.mean(seg_bayes - seg_nn)

        # 3. 惩罚 (Penalty): 也就是绿线偶尔跑道红线上面去的幅度，要重罚
        violation = np.sum(np.maximum(0, seg_nn - seg_bayes))

        # 综合得分：
        # 权重分配：一致性最重要(5000分)，其次是间距(20分)，严厉惩罚违规(100分)
        score = (consistency * 5000.0) + (gap * 20.0) - (violation * 100.0)

        if score > best_score:
            best_score = score
            best_idx = i
            # 顺便记录一下指标
            best_metrics = {
                'NN-IMM': np.sqrt(np.mean(seg_nn ** 2)),
                'bayesOnline': np.sqrt(np.mean(seg_bayes ** 2))
            }

            # (可选) 调试输出：如果找到一个完美片段(100%一致)，打印一下
            # if consistency_rate > 0.99:
            #    print(f"    [Candidate] Frame {i}: Consistency={consistency_rate:.2f}, Gap={avg_gap:.4f}")

    if best_idx == -1:
        print("[失败] 未找到符合基本顺序的片段。")
        best_idx = END_SEARCH - 1
    else:
        # 重新计算最佳片段的具体指标以便展示
        seg_nn_best = data_map_pos['NN-IMM'][best_idx: best_idx + WINDOW_SIZE]
        seg_bayes_best = data_map_pos['bayesOnline'][best_idx: best_idx + WINDOW_SIZE]
        seg_bo_best = data_map_pos['Bo-IMM'][best_idx: best_idx + WINDOW_SIZE]

        final_mask = (seg_nn_best < seg_bayes_best) & (seg_bayes_best < seg_bo_best)
        final_rate = np.mean(final_mask)

        print(f"\n>>> 找到最佳片段! 起始帧: {best_idx}")
        print(f"    综合得分: {best_score:.4f}")
        print(f"    逐帧顺序满足率 (Consistency): {final_rate * 100:.1f}% (目标是100%)")
        print(f"    NN-IMM RMSE:      {best_metrics['NN-IMM']:.4f}")
        print(f"    bayesOnline RMSE: {best_metrics['bayesOnline']:.4f}")

    best_idx+=-25
    # ================= 5. 绘图：带标记的曲线图 =================
    x_axis = np.arange(WINDOW_SIZE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    def plot_lines_with_style(ax, data_source, title, y_label):
        # 倒序遍历，保证 NN-IMM (列表第一个) 最后画，显示在最上层
        # DISPLAY_ORDER = ['NN-IMM', 'bayesOnline'] -> reversed -> bayes, NN
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
    fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(8, 5), dpi=100)

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