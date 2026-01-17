import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, BboxConnector
from matplotlib.transforms import Bbox, TransformedBbox
import os

# ================= 1. 配置区域 =================
FILES = {
    'Pos': 'result/nn_results_POS.npz',
    'Pos + Vel': 'result/nn_results_POS_VEL.npz',
    'Pos + Vel + Acc': 'result/nn_results_POS_VEL_ACC.npz'
}

# 统一参数
DT = 1 / 30
START_FRAME = 100  # 统一跳过初始化
WINDOW_SIZE = 50
MARK_EVERY = 10

# [样式配置]
# 颜色映射
COLORS = {
    'Pos + Vel + Acc': '#006400',  # 深绿 (Best)
    'Pos + Vel': '#d62728',  # 红色 (Worst)
    'Pos': '#1f77b4' # 蓝色 (Middle)

}

# 局部图样式 (Local Style) - 复刻 compare_bo_adp_bayes.py
# 包含线型 (ls) 和 标记 (mk)
LOCAL_STYLES = {
    'Pos + Vel + Acc': {'ls': '-', 'mk': '+', 'ms': 8, 'lw': 2.0, 'alpha': 0.95, 'zorder': 10},
    'Pos + Vel': {'ls': '-.', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.85, 'zorder': 5},  # Worst
    'Pos': {'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.80, 'zorder': 7}  # Middle

}

# 全局图样式 (Global Style) - 复刻 compare_bo_adp_bayes_global.py
# 仅控制粗细 (lw) 和 透明度 (alpha)，强制实线
GLOBAL_STYLES = {
    'Pos + Vel + Acc': {'lw': 1.0, 'alpha': 0.95, 'zorder': 10},
    'Pos + Vel': {'lw': 1.0, 'alpha': 0.70, 'zorder': 1},  # 最细# 最粗
    'Pos': {'lw': 1.0, 'alpha': 0.85, 'zorder': 5}  # 中等

}

# 显示顺序 (图例顺序: Best -> Middle -> Worst)
DISPLAY_ORDER = ['Pos + Vel + Acc', 'Pos + Vel', 'Pos']


def draw_combined_figure(data_dict, title_text, y_label, best_idx):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 重新计算时间轴
    min_len = min(len(v) for v in data_dict.values())
    time_axis = np.arange(min_len) * DT

    all_global_values = []
    local_max_val = 0
    zoom_start = best_idx
    zoom_end = best_idx + WINDOW_SIZE

    # 1. 绘制全局背景 (使用 GLOBAL_STYLES)
    for name in reversed(DISPLAY_ORDER):
        if name not in data_dict: continue
        full_y = data_dict[name][:min_len]
        plot_y = full_y[START_FRAME:]
        plot_x = time_axis[START_FRAME:min_len]

        all_global_values.extend(plot_y)

        if zoom_end <= len(full_y):
            local_seg = full_y[zoom_start:zoom_end]
            if len(local_seg) > 0:
                local_max_val = max(local_max_val, np.max(local_seg))

        s = GLOBAL_STYLES[name]
        c = COLORS[name]
        ax.plot(plot_x, plot_y, c=c, ls='-', lw=s['lw'], alpha=s['alpha'], zorder=s['zorder'], label=name)

    # 2. 设置 Y 轴留白 (防重叠)
    global_data_max = np.percentile(all_global_values, 99.5) if all_global_values else 1.0
    ax.set_ylim(0, global_data_max * 2.5)

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)

    # 3. 绘制悬浮子图 (使用 LOCAL_STYLES)
    axins = ax.inset_axes([0.05, 0.55, 0.45, 0.40])
    local_x = np.arange(WINDOW_SIZE)  # 子图用相对帧数
    local_vals_inset = []

    for name in reversed(DISPLAY_ORDER):
        if name not in data_dict: continue
        local_y = data_dict[name][zoom_start:zoom_end]
        local_vals_inset.extend(local_y)

        s = LOCAL_STYLES[name]
        c = COLORS[name]
        axins.plot(local_x, local_y, c=c, ls=s['ls'], lw=1.0, marker=s['mk'], ms=s['ms'], markevery=MARK_EVERY,
                   alpha=s['alpha'], zorder=s['zorder'])

    axins.set_xlim(0, WINDOW_SIZE)
    if local_vals_inset: axins.set_ylim(0, max(local_vals_inset) * 1.15)
    axins.grid(True, linestyle=':', alpha=0.5)

    # 4. 连接线与框
    box_x0 = time_axis[zoom_start]
    box_width = time_axis[zoom_end - 1] - box_x0
    box_height = local_max_val

    rect_patch = Rectangle((box_x0, 0), box_width, box_height, fill=False, edgecolor="k", linestyle="--", linewidth=0.8,
                           alpha=0.6)
    ax.add_patch(rect_patch)

    rect_bbox = Bbox.from_bounds(box_x0, 0, box_width, box_height)
    rect_transform = TransformedBbox(rect_bbox, ax.transData)

    ax.add_patch(BboxConnector(axins.bbox, rect_transform, loc1=3, loc2=2, edgecolor="k", linestyle="--", linewidth=0.8,
                               alpha=0.5))
    ax.add_patch(BboxConnector(axins.bbox, rect_transform, loc1=4, loc2=1, edgecolor="k", linestyle="--", linewidth=0.8,
                               alpha=0.5))

    return fig

def main():
    # ================= 2. 加载数据 =================
    data_map_pos = {}
    data_map_vel = {}

    print(f"=== 开始读取数据 (跳过前 {START_FRAME} 帧) ===")

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}")
            continue

        try:
            data = np.load(filename)

            # 智能识别键名
            if 'err_nn_pos' in data:
                pos = data['err_nn_pos']
                vel = data['err_nn_vel'] if 'err_nn_vel' in data else np.zeros_like(pos)
            elif 'err_pos' in data:
                pos = data['err_pos']
                vel = data['err_vel']
            else:
                # 兼容其他可能的键名
                keys = list(data.keys())
                pos_key = next((k for k in keys if 'pos' in k), None)
                vel_key = next((k for k in keys if 'vel' in k), None)
                if pos_key:
                    pos = data[pos_key]
                    vel = data[vel_key] if vel_key else np.zeros_like(pos)
                else:
                    print(f"无法识别文件结构: {filename}")
                    continue

            data_map_pos[label] = pos
            data_map_vel[label] = vel
            print(f"加载: {label:<25} | Len: {len(pos)}")

        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")

    # 校验数据
    required = ['Pos + Vel + Acc', 'Pos', 'Pos + Vel']
    if not all(k in data_map_pos for k in required):
        print("[错误] 缺少关键数据，无法继续。")
        return

    # ================= 3. 计算全局指标 =================
    global_metrics = {'pos': {}, 'vel': {}}
    min_len = min(len(v) for v in data_map_pos.values())
    time_axis_global = np.arange(min_len) * DT

    print("\n" + "=" * 60)
    print(f"{'Model':<25} | {'Pos RMSE':<10} | {'Vel RMSE':<10}")
    print("-" * 60)

    for name in DISPLAY_ORDER:
        if name not in data_map_pos: continue

        # 截取有效段
        d_pos = data_map_pos[name][START_FRAME:min_len]
        d_vel = data_map_vel[name][START_FRAME:min_len]

        rmse_p = np.sqrt(np.mean(d_pos ** 2))
        rmse_v = np.sqrt(np.mean(d_vel ** 2))

        global_metrics['pos'][name] = rmse_p
        global_metrics['vel'][name] = rmse_v

        print(f"{name:<25} | {rmse_p:<10.4f} | {rmse_v:<10.4f}")
    print("=" * 60)

    # ================= 4. 局部高光片段搜索 =================
    # 目标: All < Pos < Pos + Vel
    print(f"\n>>> 开始搜索最佳片段 (Win={WINDOW_SIZE})...")
    print(f"    筛选条件: All < Pos + Vel < Pos")

    best_score = -1.0
    best_idx = START_FRAME
    found = False

    end_search = min_len - WINDOW_SIZE

    for i in range(START_FRAME, end_search):
        # 计算窗口 RMSE
        rmses = {}
        for name in required:
            seg = data_map_pos[name][i: i + WINDOW_SIZE]
            rmses[name] = np.sqrt(np.mean(seg ** 2))

        r_all = rmses['Pos + Vel + Acc']
        r_p = rmses['Pos']
        r_pv = rmses['Pos + Vel']

        # [核心判定] All < Pos < Pos + Vel
        if r_all < r_pv < r_p:
            found = True
            # 分离度: 越开越好
            gap1 = r_pv - r_all  # Gap between Best and Middle
            gap2 = r_p - r_pv  # Gap between Middle and Worst
            score = min(gap1, gap2)

            if score > best_score:
                best_score = score
                best_idx = i

    if not found:
        print("[警告] 未找到满足严格排序的片段，显示 All 误差最小片段。")
        # Fallback
        vals = data_map_pos['Pos + Vel + Acc']
        # 滑动平均找最小
        rolling_err = np.convolve(vals[START_FRAME:] ** 2, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
        best_idx = START_FRAME + np.argmin(rolling_err)
    else:
        print(f"    [锁定] 最佳片段起始帧: {best_idx}")
        print(f"    [指标] 分离度: {best_score:.5f}")

    fig_comb_pos = draw_combined_figure(data_map_pos, 'Position Error', 'Position Error (m)', best_idx)
    fig_comb_pos.show()

    fig_comb_vel = draw_combined_figure(data_map_vel, 'Velocity Error', 'Velocity Error (m/s)', best_idx)
    fig_comb_vel.show()
    # ================= 5. 绘图: 局部对比图 (Local) =================
    # 风格参考: compare_bo_adp_bayes.py
    fig_loc, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=120, sharex=True)

    # [修改] 横坐标使用相对时间步 0 - 100
    x_loc = np.arange(WINDOW_SIZE)

    def plot_local_segment(ax, source_map, title, ylabel):
        # 倒序绘制，保证 Best 在最上层
        for name in reversed(DISPLAY_ORDER):
            if name not in source_map: continue

            y_data = source_map[name][best_idx: best_idx + WINDOW_SIZE]
            s = LOCAL_STYLES[name]  # 获取线型和Marker配置
            c = COLORS[name]

            ax.plot(x_loc, y_data,  # 使用 x_loc (0-100)
                    c=c,
                    ls=s['ls'],
                    lw=s['lw'],
                    marker=s['mk'],
                    ms=s['ms'],
                    markevery=MARK_EVERY,
                    alpha=s['alpha'],
                    zorder=s['zorder'],
                    label=name)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', framealpha=0.95, shadow=True)
        # 设置 x 轴范围 0 到 100
        ax.set_xlim(0, WINDOW_SIZE)

    plot_local_segment(ax1, data_map_pos, 'Position Error Comparison', 'Position Error (m)')
    plot_local_segment(ax2, data_map_vel, 'Velocity Error Comparison', 'Velocity Error (m/s)')

    ax2.set_xlabel('Time Step (k)', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ================= 6. 绘图: 全局轨迹图 (Global) =================
    # 风格参考: compare_bo_adp_bayes_global.py
    fig_glob = plt.figure(figsize=(12, 10))

    # 上子图: Pos
    ax_g1 = plt.subplot(2, 1, 1)

    for name in reversed(DISPLAY_ORDER):
        if name not in data_map_pos: continue

        y_data = data_map_pos[name][START_FRAME:min_len]
        t_data = time_axis_global[START_FRAME:min_len]

        s = GLOBAL_STYLES[name]  # 获取粗细配置
        c = COLORS[name]
        rmse = global_metrics['pos'][name]

        # 强制实线，无Marker
        ax_g1.plot(t_data, y_data,
                   c=c, ls='-',
                   lw=s['lw'], alpha=s['alpha'], zorder=s['zorder'],
                   label=f'{name}')

    ax_g1.set_title('Global Position Error Trajectory', fontsize=12, fontweight='bold')
    ax_g1.set_ylabel('Position Error (m)')
    ax_g1.grid(True, linestyle='--', alpha=0.4)
    ax_g1.legend(loc='upper right', framealpha=0.9, shadow=True)

    # 下子图: Vel
    ax_g2 = plt.subplot(2, 1, 2)

    for name in reversed(DISPLAY_ORDER):
        if name not in data_map_vel: continue

        y_data = data_map_vel[name][START_FRAME:min_len]
        t_data = time_axis_global[START_FRAME:min_len]

        s = GLOBAL_STYLES[name]
        c = COLORS[name]
        rmse = global_metrics['vel'][name]

        ax_g2.plot(t_data, y_data,
                   c=c, ls='-',
                   lw=s['lw'], alpha=s['alpha'], zorder=s['zorder'],
                   label=f'{name}')

    ax_g2.set_title('Global Velocity Error Trajectory', fontsize=12, fontweight='bold')
    ax_g2.set_ylabel('Velocity Error (m/s)')
    ax_g2.set_xlabel('Time (s)')
    ax_g2.grid(True, linestyle='--', alpha=0.4)
    ax_g2.legend(loc='upper right', framealpha=0.9, shadow=True)

    plt.tight_layout()
    plt.show()

    # ================= 7. 绘图: 柱状图 (Bar) =================
    fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(10, 5))

    bar_names = DISPLAY_ORDER
    x_pos = np.arange(len(bar_names))
    width = 0.5
    b_colors = [COLORS[n] for n in bar_names]

    # Pos Bar
    vals_p = [global_metrics['pos'][n] for n in bar_names]
    bars1 = ax_b1.bar(x_pos, vals_p, width, color=b_colors, alpha=0.85)
    ax_b1.set_title('Global Position RMSE Comparison', fontsize=12, fontweight='bold')
    ax_b1.set_ylabel('RMSE (m)')
    ax_b1.set_xticks(x_pos)
    ax_b1.set_xticklabels(bar_names, rotation=15)
    ax_b1.grid(axis='y', linestyle='--', alpha=0.4)

    # 动态 Y 轴
    if vals_p:
        mn, mx = min(vals_p), max(vals_p)
        margin = (mx - mn) * 0.5
        ax_b1.set_ylim(max(0, mn - margin) if mn > 0.5 * mx else 0, mx + margin)

        for bar in bars1:
            h = bar.get_height()
            ax_b1.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.4f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold')

    # Vel Bar
    vals_v = [global_metrics['vel'][n] for n in bar_names]
    bars2 = ax_b2.bar(x_pos, vals_v, width, color=b_colors, alpha=0.85)
    ax_b2.set_title('Global Velocity RMSE Comparison', fontsize=12, fontweight='bold')
    ax_b2.set_ylabel('RMSE (m/s)')
    ax_b2.set_xticks(x_pos)
    ax_b2.set_xticklabels(bar_names, rotation=15)
    ax_b2.grid(axis='y', linestyle='--', alpha=0.4)

    if vals_v:
        mn, mx = min(vals_v), max(vals_v)
        margin = (mx - mn) * 0.5
        ax_b2.set_ylim(max(0, mn - margin) if mn > 0.5 * mx else 0, mx + margin)

        for bar in bars2:
            h = bar.get_height()
            ax_b2.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.4f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()