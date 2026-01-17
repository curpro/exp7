import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, BboxConnector
from matplotlib.transforms import Bbox, TransformedBbox
import os

# ================= 1. 配置区域 =================
FILES = {
    'OPT 10': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\Win__10\nn_results_opt10.npz',
    'OPT 40': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\Win__40\nn_results_opt40.npz',
    'OPT 20': r'D:\AFS\lunwen\lunwen1\chapter5\network\MLP\nn_results_win90.npz'
}

# 绘图顺序: 最好的放最后 (OPT 20)，保证图层在最上方
# 顺序: Worst (40) -> Mid (10) -> Best (20)
DISPLAY_ORDER = ['OPT 40', 'OPT 10', 'OPT 20']

# --- 样式配置 ---
# Global: 大图样式 (透明度较高，无标记，看整体趋势)
STYLE_GLOBAL = {
    'OPT 20': {'c': '#006400', 'lw': 1.8, 'alpha': 0.95, 'zorder': 10, 'label': 'OPT 20 (Best)'},
    'OPT 10': {'c': '#1f77b4', 'lw': 1.2, 'alpha': 0.85, 'zorder': 5, 'label': 'OPT 10'},
    'OPT 40': {'c': '#d62728', 'lw': 1.0, 'alpha': 0.70, 'zorder': 1, 'label': 'OPT 40'}
}

# Local: 子图样式 (带标记，不透明，看细节)
STYLE_LOCAL = {
    'OPT 20': {'mk': '+', 'ms': 8, 'ls': '-'},
    'OPT 10': {'mk': 'o', 'ms': 6, 'ls': '--'},
    'OPT 40': {'mk': '^', 'ms': 6, 'ls': ':'}
}

# 参数
UNIFIED_START_FRAME = 90
WINDOW_SIZE = 100
START_SEARCH = 90
MARK_EVERY = 8
MANUAL_OFFSET = 30  # 保留原文件中的手动偏移


# ================= 2. 数据处理 =================

def load_all_data():
    data_pos = {}
    data_vel = {}

    print(">>> 正在加载 NN Opt Interval 对比数据...")
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"[警告] 文件不存在: {fname}")
            continue
        try:
            raw = np.load(fname)
            p_arr, v_arr = None, None

            if 'err_nn_pos' in raw:
                p_arr = raw['err_nn_pos']
                if 'err_nn_vel' in raw:
                    v_arr = raw['err_nn_vel']
                else:
                    v_arr = np.zeros_like(p_arr)
            elif 'err_pos' in raw:
                p_arr = raw['err_pos']
                v_arr = raw['err_vel']

            if p_arr is not None:
                data_pos[name] = p_arr
                data_vel[name] = v_arr
                print(f"    Loaded {name}: {len(p_arr)} frames")
        except Exception as e:
            print(f"[Error] {name}: {e}")

    return data_pos, data_vel


def find_best_window(data_map):
    """
    寻找满足 OPT 20 < OPT 10 < OPT 40 的最佳区间
    """
    if len(data_map) < 3: return START_SEARCH

    min_len = min([len(v) for v in data_map.values()])
    end_search = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = START_SEARCH

    print("\n>>> 正在搜索最佳展示区间 (OPT 20 < 10 < 40)...")
    for i in range(START_SEARCH, end_search):
        rmses = {k: np.sqrt(np.mean(v[i:i + WINDOW_SIZE] ** 2)) for k, v in data_map.items()}

        if ('OPT 20' in rmses and 'OPT 10' in rmses and 'OPT 40' in rmses):
            # 核心判断: 20 < 10 < 40
            if (rmses['OPT 20'] < rmses['OPT 10'] < rmses['OPT 40']):
                gap1 = rmses['OPT 10'] - rmses['OPT 20']
                gap2 = rmses['OPT 40'] - rmses['OPT 10']

                score = min(gap1, gap2)

                if score > best_score:
                    best_score = score
                    best_idx = i

    # [关键] 应用原文件中的手动偏移
    print(f"    原始最佳帧: {best_idx} (Score: {best_score:.5f})")
    final_idx = best_idx + MANUAL_OFFSET
    print(f"    应用偏移后起始帧: {final_idx} (+{MANUAL_OFFSET})")

    return final_idx


# ================= 3. 绘图核心逻辑 =================

def draw_single_figure(data_dict, title_text, y_label, best_idx):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 1. 准备数据
    max_len = max([len(v) for v in data_dict.values()])
    time_axis = np.arange(max_len)
    all_global_values = []

    local_max_val = 0
    zoom_start = best_idx
    zoom_end = best_idx + WINDOW_SIZE

    # 2. 绘制全局曲线
    for name in DISPLAY_ORDER:
        if name not in data_dict: continue
        full_y = data_dict[name]

        plot_y = full_y[UNIFIED_START_FRAME:]
        plot_x = time_axis[UNIFIED_START_FRAME: UNIFIED_START_FRAME + len(plot_y)]
        all_global_values.extend(plot_y)

        if zoom_end <= len(full_y):
            local_segment = full_y[zoom_start:zoom_end]
            if len(local_segment) > 0:
                local_max_val = max(local_max_val, np.max(local_segment))

        s = STYLE_GLOBAL[name]
        ax.plot(plot_x, plot_y,
                color=s['c'], linewidth=s['lw'], alpha=s['alpha'],
                zorder=s['zorder'], label=s['label'])

    # 3. 设置留白防重叠
    global_data_max = np.percentile(all_global_values, 99.5) if all_global_values else 1.0
    final_ylim = global_data_max * 2.5
    ax.set_ylim(0, final_ylim)

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step (k)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)

    # 4. 绘制悬浮子图
    axins = ax.inset_axes([0.05, 0.55, 0.45, 0.40])

    local_vals_for_inset = []

    for name in DISPLAY_ORDER:
        if name not in data_dict: continue
        full_y = data_dict[name]

        curr_zoom_end = min(zoom_end, len(full_y))
        local_y = full_y[zoom_start:curr_zoom_end]
        local_x = time_axis[zoom_start:curr_zoom_end]
        local_vals_for_inset.extend(local_y)

        gs = STYLE_GLOBAL[name]
        ls = STYLE_LOCAL[name]

        axins.plot(local_x, local_y,
                   color=gs['c'], linestyle=ls['ls'], linewidth=1.0,
                   marker=ls['mk'], markersize=ls['ms'], markevery=MARK_EVERY,
                   alpha=0.9, zorder=gs['zorder'])

    axins.set_xlim(time_axis[zoom_start], time_axis[zoom_end - 1])
    if local_vals_for_inset:
        axins.set_ylim(0, max(local_vals_for_inset) * 1.15)
    axins.grid(True, linestyle=':', alpha=0.5)

    # 5. 绘制矩形框与连接线
    box_x0 = time_axis[zoom_start]
    box_width = time_axis[zoom_end - 1] - box_x0
    box_height = local_max_val

    rect_patch = Rectangle((box_x0, 0), box_width, box_height,
                           fill=False, edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.add_patch(rect_patch)

    rect_bbox_data = Bbox.from_bounds(box_x0, 0, box_width, box_height)
    rect_bbox_display = TransformedBbox(rect_bbox_data, ax.transData)

    p1 = BboxConnector(axins.bbox, rect_bbox_display, loc1=3, loc2=2,
                       edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.add_patch(p1)

    p2 = BboxConnector(axins.bbox, rect_bbox_display, loc1=4, loc2=1,
                       edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.add_patch(p2)

    return fig


# ================= 4. 主程序 =================

def main():
    d_pos, d_vel = load_all_data()

    if not d_pos:
        print("[错误] 未加载到数据")
        return

    best_idx = find_best_window(d_pos)

    print(">>> 正在生成图表 1: 位置误差...")
    fig1 = draw_single_figure(d_pos, 'Position Error Comparison', 'Position Error (m)', best_idx)
    fig1.show()

    if d_vel and len(d_vel) > 0:
        print(">>> 正在生成图表 2: 速度误差...")
        fig2 = draw_single_figure(d_vel, 'Velocity Error Comparison', 'Velocity Error (m/s)', best_idx)
        fig2.show()

    plt.show()


if __name__ == "__main__":
    main()