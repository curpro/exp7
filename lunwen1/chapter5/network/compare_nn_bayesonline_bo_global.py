import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'bayesOnline': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\imm_results_90.npz',
    'NN-IMM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\traj1\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
    'Bo-IMM': r'D:\AFS\lunwen\lunwen1\newPy\res_data\bo_imm_results.npz'
}

DT = 1 / 30
UNIFIED_START_FRAME = 90  # 统一跳过初始化帧数

# 风格配置 (基于 compare_mlp_bayesonline，但去掉了 Marker 以适应全局显示)
STYLE_CONFIG = {
    # NN-IMM (Best): 深绿, 实线 -> 主角
    'NN-IMM': {
        'c': '#006400', 'ls': '-', 'lw': 1.8, 'alpha': 0.95, 'zorder': 10,
        'label': 'NN-IMM'
    },

    # bayesOnline (Baseline): 蓝色, 虚线 -> 对比项
    'bayesOnline': {
        'c': '#d62728', 'ls': '-', 'lw': 1.2, 'alpha': 0.85, 'zorder': 7,
        'label': 'bayesOnline'
    },

    # Bo-IMM
    'Bo-IMM': {
        'c': '#1f77b4', 'ls': '-', 'lw': 1.0, 'alpha': 0.70, 'zorder': 5,
        'label': 'Bo-IMM'
    }
}

# 显示顺序
DISPLAY_ORDER = ['NN-IMM', 'bayesOnline', 'Bo-IMM']


# ===============================================

def load_and_compare_global():
    results = {}
    time_axis = None

    print(f"=== 开始读取全局对比数据 (统一跳过前 {UNIFIED_START_FRAME} 帧) ===")

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}")
            continue

        try:
            data = np.load(filename)

            # --- 智能识别键名 ---
            # 1. 优先尝试 NN 格式
            if 'err_nn_pos' in data:
                err_pos = data['err_nn_pos']
                if 'err_nn_vel' in data:
                    err_vel = data['err_nn_vel']
                else:
                    err_vel = np.zeros_like(err_pos)

            # 2. [新增] 适配 bayesOnline 的 'err_online_pos'
            elif 'err_online_pos' in data:
                print(f"    [{label}] 识别到 'err_online_pos'，正在读取...")
                err_pos = data['err_online_pos']
                err_vel = data['err_online_vel']

            # 3. 兼容通用格式
            elif 'err_pos' in data:
                err_pos = data['err_pos']
                err_vel = data['err_vel']

            # 4. 兼容 Bo/Adp 格式
            elif 'err_bo_pos' in data:
                err_pos = data['err_bo_pos']
                err_vel = data['err_bo_vel']
            elif 'err_adp_pos' in data:
                err_pos = data['err_adp_pos']
                err_vel = data['err_adp_vel']

            else:
                print(f"[错误] {label} 文件中未找到预期的误差数据键名。")
                continue

            # 获取或生成时间轴
            if time_axis is None:
                if 't' in data:
                    time_axis = data['t']
                else:
                    time_axis = np.arange(len(err_pos)) * DT

            # 截取有效数据 (跳过初始化阶段)
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
            print(f"加载: {label:<15} | Pos RMSE: {rmse_pos:.4f}")

        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")

    if not results:
        print("[错误] 未加载到任何数据。")
        return

    # ================= 打印统计表格 =================
    print("\n" + "=" * 90)
    print(f"{'Model Name':<20} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15} | {'Pos Var':<15}")
    print("-" * 90)

    # 过滤掉没加载成功的文件
    valid_order = [k for k in DISPLAY_ORDER if k in results]

    vals_pos = []
    vals_vel = []
    labels = []

    for key in valid_order:
        r = results[key]
        print(f"{key:<20} | {r['rmse_pos']:<15.4f} | {r['rmse_vel']:<15.4f} | {r['var_pos']:<15.4f}")
        vals_pos.append(r['rmse_pos'])
        vals_vel.append(r['rmse_vel'])
        labels.append(key)
    print("=" * 90 + "\n")

    # ================= 绘图 1: 柱状图 (Global RMSE) =================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.5

    # 获取对应颜色
    bar_colors = [STYLE_CONFIG[label]['c'] for label in labels]

    # --- 左图：位置 RMSE ---
    bars1 = ax1.bar(x, vals_pos, width, color=bar_colors, alpha=0.85)
    ax1.set_title(f'Global Position RMSE', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # 动态缩放 Y 轴
    if len(vals_pos) > 0:
        min_p, max_p = min(vals_pos), max(vals_pos)
        margin_p = (max_p - min_p) * 0.8 if max_p != min_p else 0.1
        ax1.set_ylim(max(0, min_p - margin_p) if min_p > 1.0 else 0, max_p + margin_p)

    # --- 右图：速度 RMSE ---
    bars2 = ax2.bar(x, vals_vel, width, color=bar_colors, alpha=0.85)
    ax2.set_title(f'Global Velocity RMSE', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    if len(vals_vel) > 0:
        min_v, max_v = min(vals_vel), max(vals_vel)
        margin_v = (max_v - min_v) * 0.8 if max_v != min_v else 0.1
        ax2.set_ylim(max(0, min_v - margin_v) if min_v > 1.0 else 0, max_v + margin_v)

    # 数值标注
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    autolabel(bars1, ax1)
    autolabel(bars2, ax2)
    fig1.tight_layout()

    # ================= 绘图 2: 全程曲线图 (Global Trajectory) =================
    plt.figure(figsize=(12, 10))

    # --- 上图：位置误差 ---
    plt.subplot(2, 1, 1)

    # 倒序遍历，保证 NN-IMM (主角) 最后画
    for key in reversed(valid_order):
        err = results[key]['err_pos_seq']
        rmse = results[key]['rmse_pos']

        # 截取数据
        plot_data = err[UNIFIED_START_FRAME:]
        plot_time = time_axis[UNIFIED_START_FRAME:]

        # 对齐长度
        min_len_plot = min(len(plot_data), len(plot_time))
        plot_data = plot_data[:min_len_plot]
        plot_time = plot_time[:min_len_plot]

        s = STYLE_CONFIG[key]

        plt.plot(plot_time, plot_data,
                 label=f"{s['label']} (RMSE={rmse:.2f}m)",
                 color=s['c'],
                 linestyle=s['ls'],
                 linewidth=s['lw'],
                 alpha=s['alpha'],
                 zorder=s['zorder'])

    plt.title(f'Global Position Error Trajectory', fontsize=12, fontweight='bold')
    plt.ylabel('Position Error (m)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    # --- 下图：速度误差 ---
    plt.subplot(2, 1, 2)

    for key in reversed(valid_order):
        err = results[key]['err_vel_seq']
        rmse = results[key]['rmse_vel']

        plot_data = err[UNIFIED_START_FRAME:]
        plot_time = time_axis[UNIFIED_START_FRAME:]

        min_len_plot = min(len(plot_data), len(plot_time))
        plot_data = plot_data[:min_len_plot]
        plot_time = plot_time[:min_len_plot]

        s = STYLE_CONFIG[key]

        plt.plot(plot_time, plot_data,
                 label=f"{s['label']} (RMSE={rmse:.2f}m/s)",
                 color=s['c'],
                 linestyle=s['ls'],
                 linewidth=s['lw'],
                 alpha=s['alpha'],
                 zorder=s['zorder'])

    plt.title(f'Global Velocity Error Trajectory', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity Error (m/s)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_compare_global()