import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区域 =================
FILES = {
    'bayesOnline': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\traj3\imm_results_90.npz',
    'adpt_imm': r'D:\AFS\lunwen\lunwen1\newPy\res_data\traj3\adp_results.npz',
    'Bo-IMM': r'D:\AFS\lunwen\lunwen1\newPy\res_data\traj3\bo_imm_results.npz'
}

DT = 1 / 30
UNIFIED_START_FRAME = 90  # 统一跳过初始化帧数


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

            # --- 智能识别键名 (修改了这里) ---
            if 'err_adp_pos' in data:
                # 适配 adpt_imm
                err_pos = data['err_adp_pos']
                err_vel = data['err_adp_vel']
            elif 'err_bo_pos' in data:
                # 适配 Bo-IMM
                err_pos = data['err_bo_pos']
                err_vel = data['err_bo_vel']

            # === [核心修改] 加上这一段来读 bayesOnline ===
            elif 'err_online_pos' in data:
                err_pos = data['err_online_pos']
                err_vel = data['err_online_vel']
            # ==========================================

            elif 'err_nn_pos' in data:
                # 适配旧版 NN
                err_pos = data['err_nn_pos']
                # 部分旧文件可能没有 velocity，做防错处理
                if 'err_nn_vel' in data:
                    err_vel = data['err_nn_vel']
                else:
                    err_vel = np.zeros_like(err_pos)
            else:
                # 尝试通用键名
                err_pos = data['err_pos']
                err_vel = data['err_vel']

            # 获取时间轴 (只需要获取一次)
            if time_axis is None:
                if 't' in data:
                    time_axis = data['t']
                else:
                    # 如果没有时间轴，自动生成
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
    print(f"{'Model Name':<15} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15} | {'Pos Var':<15}")
    print("-" * 90)

    # 指定显示顺序
    display_order = ['bayesOnline', 'adpt_imm', 'Bo-IMM']
    # 过滤掉没加载成功的文件
    display_order = [k for k in display_order if k in results]

    vals_pos = []
    vals_vel = []
    labels = []

    for key in display_order:
        r = results[key]
        print(f"{key:<15} | {r['rmse_pos']:<15.4f} | {r['rmse_vel']:<15.4f} | {r['var_pos']:<15.4f}")
        vals_pos.append(r['rmse_pos'])
        vals_vel.append(r['rmse_vel'])
        labels.append(key)
    print("=" * 90 + "\n")

    # === [配置] 颜色与样式 (保持与局部图一致的色系) ===
    color_map = {
        'bayesOnline': '#006400',  # 深绿 (Best)
        'adpt_imm': '#d62728',  # 红色 (Middle)
        'Bo-IMM': '#1f77b4'  # 蓝色 (Worst)
    }

    # 全局绘图样式：
    # 相比局部图，这里去掉了 Marker，改用不同粗细/透明度来区分
    style_map = {
        'bayesOnline': {'lw': 1.8, 'alpha': 0.95, 'zorder': 10},  # 最粗、最实、最上层
        'adpt_imm': {'lw': 1.2, 'alpha': 0.85, 'zorder': 5},  # 中等
        'Bo-IMM': {'lw': 1.0, 'alpha': 0.70, 'zorder': 1}  # 最细、较透、底层
    }

    # ================= 绘图 1: 柱状图 (Bar Chart) =================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.5
    bar_colors = [color_map[label] for label in labels]

    # --- 左图：位置 RMSE 柱状图 ---
    bars1 = ax1.bar(x, vals_pos, width, color=bar_colors, alpha=0.85)
    ax1.set_title(f'Global Position RMSE Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # 动态缩放 Y 轴 (突出差异)
    if len(vals_pos) > 0:
        min_p, max_p = min(vals_pos), max(vals_pos)
        margin_p = (max_p - min_p) * 0.8 if max_p != min_p else 0.1
        # 如果差异很小，不要把底部切掉太多
        ax1.set_ylim(max(0, min_p - margin_p) if min_p > 2 else 0, max_p + margin_p)

    # --- 右图：速度 RMSE 柱状图 ---
    bars2 = ax2.bar(x, vals_vel, width, color=bar_colors, alpha=0.85)
    ax2.set_title(f'Global Velocity RMSE Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    if len(vals_vel) > 0:
        min_v, max_v = min(vals_vel), max(vals_vel)
        margin_v = (max_v - min_v) * 0.8 if max_v != min_v else 0.1
        ax2.set_ylim(max(0, min_v - margin_v) if min_v > 2 else 0, max_v + margin_v)

    # 标注数值
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

    # ================= 绘图 2: 全程曲线图 (Time Series) =================
    plt.figure(figsize=(12, 10))

    # --- 上图：位置误差 ---
    plt.subplot(2, 1, 1)

    # 倒序遍历，保证 bayesOnline (第一个) 最后画，显示在最上层
    for key in reversed(display_order):
        err = results[key]['err_pos_seq']
        rmse = results[key]['rmse_pos']

        # 截取显示部分
        plot_data = err[UNIFIED_START_FRAME:]
        plot_time = time_axis[UNIFIED_START_FRAME:]

        # 确保时间轴长度对齐
        if len(plot_time) > len(plot_data):
            plot_time = plot_time[:len(plot_data)]
        elif len(plot_data) > len(plot_time):
            plot_data = plot_data[:len(plot_time)]

        style = style_map.get(key, {'lw': 1, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        plt.plot(plot_time, plot_data,
                 label=f'{key} (RMSE={rmse:.2f}m)',
                 color=line_color,
                 linewidth=style['lw'],
                 alpha=style['alpha'],
                 zorder=style['zorder'])

    plt.title(f'Global Position Error Trajectory', fontsize=12, fontweight='bold')
    plt.ylabel('Position Error (m)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    # 设置合理的显示范围，防止个别离群点破坏视图
    # plt.ylim(0, np.percentile(vals_pos, 95) * 3)

    # --- 下图：速度误差 ---
    plt.subplot(2, 1, 2)

    for key in reversed(display_order):
        err = results[key]['err_vel_seq']
        rmse = results[key]['rmse_vel']

        plot_data = err[UNIFIED_START_FRAME:]
        plot_time = time_axis[UNIFIED_START_FRAME:]

        if len(plot_time) > len(plot_data):
            plot_time = plot_time[:len(plot_data)]
        elif len(plot_data) > len(plot_time):
            plot_data = plot_data[:len(plot_time)]

        style = style_map.get(key, {'lw': 1, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        plt.plot(plot_time, plot_data,
                 label=f'{key} (RMSE={rmse:.2f}m/s)',
                 color=line_color,
                 linewidth=style['lw'],
                 alpha=style['alpha'],
                 zorder=style['zorder'])

    plt.title(f'Global Velocity Error Trajectory', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity Error (m/s)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_compare_global()