import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
# 保持您想要的简洁标签：Win 90, Win 180, Win 45
FILES = {
    'Win 45': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_45\r_300_20\imm_results_45.npz',
    'Win 180': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_180\r_300_20\imm_results_180.npz',
    'Win 90': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\imm_results_90.npz'
}

DT = 1 / 30
UNIFIED_START_FRAME = 90


# ======================================

def load_and_compare_nn():
    results = {}
    time_axis = None

    print(f"=== 开始读取 NN 对比数据 (统一跳过前 {UNIFIED_START_FRAME} 帧) ===")

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}")
            continue

        try:
            data = np.load(filename)
            err_pos = data['err_online_pos']
            err_vel = data['err_online_vel']
            t = data['t']

            if time_axis is None:
                time_axis = t

            safe_start = min(UNIFIED_START_FRAME, len(err_pos) - 1)

            rmse_pos = np.sqrt(np.mean(err_pos[safe_start:] ** 2))
            rmse_vel = np.sqrt(np.mean(err_vel[safe_start:] ** 2))
            var_pos = np.var(err_pos[safe_start:])

            results[label] = {
                'rmse_pos': rmse_pos,
                'rmse_vel': rmse_vel,
                'var_pos': var_pos,
                'err_pos_seq': err_pos,
                'err_vel_seq': err_vel
            }
            print(f"加载: {label:<12} | Pos RMSE: {rmse_pos:.4f}")

        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")

    if not results:
        return

    # ================= 打印表格 =================
    print("\n" + "=" * 80)
    print(f"{'Window Size':<15} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15} | {'Pos Var':<15}")
    print("-" * 80)

    # 排序
    sorted_keys = ['Win 90', 'Win 180', 'Win 45']
    sorted_keys = [k for k in sorted_keys if k in results]

    vals_pos = []
    vals_vel = []
    labels = []

    for key in sorted_keys:
        r = results[key]
        print(f"{key:<15} | {r['rmse_pos']:<15.4f} | {r['rmse_vel']:<15.4f} | {r['var_pos']:<15.4f}")
        vals_pos.append(r['rmse_pos'])
        vals_vel.append(r['rmse_vel'])
        labels.append(key)
    print("=" * 80 + "\n")

    # === [配置] 颜色与样式 ===
    color_map = {
        'Win 90': '#006400',  # 绿
        'Win 180': '#1f77b4',  # 蓝
        'Win 45': '#d62728'  # 红
    }

    # [已恢复] 去掉了 'ls' (线型)，全部默认为实线
    # 依然保留粗细和透明度的区别，保证 Win 90 看起来最突出
    style_map = {
        'Win 90': {'lw': 1.0, 'alpha': 0.80, 'zorder': 10},  # 最粗、最实
        'Win 180': {'lw': 1.5, 'alpha': 0.80, 'zorder': 5},
        'Win 45': {'lw': 1.0, 'alpha': 0.80, 'zorder': 1}  # 最细、最淡
    }

    # ================= 绘图 1: 柱状图 =================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.5
    bar_colors = [color_map[label] for label in labels]

    # 左图：位置
    bars1 = ax1.bar(x, vals_pos, width, color=bar_colors, alpha=0.85)
    ax1.set_title(f'Position RMSE Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # 柱状图依然保留动态缩放，因为柱状图不涉及尖峰问题，只看平均值
    min_p, max_p = min(vals_pos), max(vals_pos)
    margin_p = (max_p - min_p) * 0.8 if max_p != min_p else 0.1
    ax1.set_ylim(max(0, min_p - margin_p), max_p + margin_p)

    # 右图：速度
    bars2 = ax2.bar(x, vals_vel, width, color=bar_colors, alpha=0.85)
    ax2.set_title(f'Velocity RMSE Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    min_v, max_v = min(vals_vel), max(vals_vel)
    margin_v = (max_v - min_v) * 0.8 if max_v != min_v else 0.1
    ax2.set_ylim(max(0, min_v - margin_v), max_v + margin_v)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(bars1, ax1)
    autolabel(bars2, ax2)
    fig1.tight_layout()

    # ================= 绘图 2: 原始曲线图 (恢复全范围) =================
    plt.figure(figsize=(12, 10))

    # --- 上图：位置误差 ---
    plt.subplot(2, 1, 1)

    for key in sorted_keys:
        err = results[key]['err_pos_seq']
        rmse = results[key]['rmse_pos']

        plot_data = err[UNIFIED_START_FRAME:]

        style = style_map.get(key, {'lw': 1, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        plt.plot(time_axis[UNIFIED_START_FRAME:], plot_data,
                 label=f'{key} (RMSE={rmse:.2f}m)',
                 color=line_color,
                 linewidth=style['lw'],
                 alpha=style['alpha'],
                 zorder=style['zorder'])
        # [已恢复] 不再设置 linestyle，默认实线

    plt.title(f'Position Error')
    plt.ylabel('Position Error (m)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    # --- 下图：速度误差 ---
    plt.subplot(2, 1, 2)

    for key in sorted_keys:
        err = results[key]['err_vel_seq']
        rmse = results[key]['rmse_vel']

        plot_data = err[UNIFIED_START_FRAME:]

        style = style_map.get(key, {'lw': 1, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        plt.plot(time_axis[UNIFIED_START_FRAME:], plot_data,
                 label=f'{key} (RMSE={rmse:.2f}m/s)',
                 color=line_color,
                 linewidth=style['lw'],
                 alpha=style['alpha'],
                 zorder=style['zorder'])

    plt.title(f'Velocity Error')
    plt.ylabel('Velocity Error (m/s)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_compare_nn()