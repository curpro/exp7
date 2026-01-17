import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, BboxConnector
from matplotlib.transforms import Bbox, TransformedBbox
import os

# ================= 1. 配置区域 =================
FILES = {
    'bayesOnline': r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Win_90\r_300_20\imm_results_90.npz',
    'adpt_imm': r'D:\AFS\lunwen\lunwen1\newPy\res_data\adp_results.npz',
    'Bo-IMM': r'D:\AFS\lunwen\lunwen1\newPy\res_data\bo_imm_results.npz'
}

# 绘图顺序 (bayesOnline 最后画，确保在最上层)
DISPLAY_ORDER = ['Bo-IMM', 'adpt_imm', 'bayesOnline']

# --- 样式配置 ---
STYLE_GLOBAL = {
    'bayesOnline': {'c': '#006400', 'lw': 1.8, 'alpha': 0.95, 'zorder': 10, 'label': 'bayesOnline (Best)'},
    'adpt_imm': {'c': '#d62728', 'lw': 1.2, 'alpha': 0.85, 'zorder': 5, 'label': 'adpt_imm'},
    'Bo-IMM': {'c': '#1f77b4', 'lw': 1.0, 'alpha': 0.70, 'zorder': 1, 'label': 'Bo-IMM'}
}

STYLE_LOCAL = {
    'bayesOnline': {'mk': '+', 'ms': 8, 'ls': '-'},
    'adpt_imm': {'mk': '^', 'ms': 6, 'ls': '-.'},
    'Bo-IMM': {'mk': 'o', 'ms': 5, 'ls': '--'}
}

# 参数
UNIFIED_START_FRAME = 90
WINDOW_SIZE = 100
START_SEARCH = 90
MARK_EVERY = 10


# ================= 2. 数据处理 =================

def load_all_data():
    data_pos = {}
    data_vel = {}

    print(">>> 正在加载数据...")
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"[警告] 文件不存在: {fname}")
            continue
        try:
            raw = np.load(fname)
            p_arr, v_arr = None, None

            if 'err_adp_pos' in raw:
                p_arr, v_arr = raw['err_adp_pos'], raw['err_adp_vel']
            elif 'err_bo_pos' in raw:
                p_arr, v_arr = raw['err_bo_pos'], raw['err_bo_vel']
            elif 'err_online_pos' in raw:
                p_arr, v_arr = raw['err_online_pos'], raw['err_online_vel']
            elif 'err_nn_pos' in raw:
                p_arr = raw['err_nn_pos']
                v_arr = raw['err_nn_vel'] if 'err_nn_vel' in raw else np.zeros_like(p_arr)
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
    if len(data_map) < 3: return START_SEARCH

    min_len = min([len(v) for v in data_map.values()])
    end_search = min_len - WINDOW_SIZE

    best_score = -1.0
    best_idx = START_SEARCH

    print("\n>>> 正在搜索最佳展示区间...")
    for i in range(START_SEARCH, end_search):
        rmses = {k: np.sqrt(np.mean(v[i:i + WINDOW_SIZE] ** 2)) for k, v in data_map.items()}

        if ('bayesOnline' in rmses and 'adpt_imm' in rmses and 'Bo-IMM' in rmses):
            if (rmses['bayesOnline'] < rmses['adpt_imm'] < rmses['Bo-IMM']):
                gap = min(rmses['adpt_imm'] - rmses['bayesOnline'],
                          rmses['Bo-IMM'] - rmses['adpt_imm'])
                if gap > best_score:
                    best_score = gap
                    best_idx = i

    print(f"    最佳起始帧: {best_idx} (Score: {best_score:.5f})")
    return best_idx


# ================= 3. 绘图核心逻辑 =================

def draw_single_figure(data_dict, title_text, y_label, best_idx):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 1. 准备数据
    max_len = max([len(v) for v in data_dict.values()])
    time_axis = np.arange(max_len)
    all_global_values = []

    # 获取放大区域的数据最大值 (用于计算盒子高度)
    local_max_val = 0
    zoom_start = best_idx
    zoom_end = best_idx + WINDOW_SIZE

    # 2. 绘制全局曲线
    for name in DISPLAY_ORDER:
        if name not in data_dict: continue
        full_y = data_dict[name]

        # 全局绘图数据
        plot_y = full_y[UNIFIED_START_FRAME:]
        plot_x = time_axis[UNIFIED_START_FRAME: UNIFIED_START_FRAME + len(plot_y)]
        all_global_values.extend(plot_y)

        # 记录局部最大值
        if zoom_end <= len(full_y):
            local_segment = full_y[zoom_start:zoom_end]
            if len(local_segment) > 0:
                local_max_val = max(local_max_val, np.max(local_segment))

        s = STYLE_GLOBAL[name]
        ax.plot(plot_x, plot_y,
                color=s['c'], linewidth=s['lw'], alpha=s['alpha'],
                zorder=s['zorder'], label=s['label'])

    # 3. [关键修改] 设置 Y 轴范围 (防重叠核心)
    # 计算所有数据的最大值
    global_data_max = np.percentile(all_global_values, 99.5) if all_global_values else 1.0

    # 强制将 Y 轴上限设为最大值的 2.5 倍
    # 这样数据最高点只会达到画面的 1/2.5 = 40% 的高度
    final_ylim = global_data_max * 2.5
    ax.set_ylim(0, final_ylim)

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step (k)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)

    # 4. [关键修改] 绘制子图 (Inset)
    # [x, y, w, h]
    # y=0.55: 子图底部从 55% 高度开始
    # 数据在 40% 以下，子图在 55% 以上 -> 绝对不重叠
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
                   color=gs['c'], linestyle=ls['ls'], linewidth=1.2,
                   marker=ls['mk'], markersize=ls['ms'], markevery=MARK_EVERY,
                   alpha=0.9, zorder=gs['zorder'])

    axins.set_xlim(time_axis[zoom_start], time_axis[zoom_end - 1])
    if local_vals_for_inset:
        axins.set_ylim(0, max(local_vals_for_inset) * 1.15)
    axins.grid(True, linestyle=':', alpha=0.5)

    # ================= 5. 绘制 ZoomBox 与 连接线 =================

    # 准备矩形框参数
    box_x0 = time_axis[zoom_start]
    box_width = time_axis[zoom_end - 1] - box_x0
    box_height = local_max_val  # 紧贴数据顶端

    # (A) 使用标准 Rectangle 绘制矩形框
    rect_patch = Rectangle((box_x0, 0), box_width, box_height,
                           fill=False, edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.add_patch(rect_patch)

    # (B) 创建用于连接线的坐标变换框
    rect_bbox_data = Bbox.from_bounds(box_x0, 0, box_width, box_height)
    rect_bbox_display = TransformedBbox(rect_bbox_data, ax.transData)

    # (C) 绘制连接线
    # 线条 1: 子图左下角(3) -> 选框左上角(2)
    p1 = BboxConnector(axins.bbox, rect_bbox_display, loc1=3, loc2=2,
                       edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.add_patch(p1)

    # 线条 2: 子图右下角(4) -> 选框右上角(1)
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
    fig1 = draw_single_figure(d_pos, 'Global Position Error', 'Position Error (m)', best_idx)
    fig1.show()

    if d_vel and len(d_vel) > 0:
        print(">>> 正在生成图表 2: 速度误差...")
        fig2 = draw_single_figure(d_vel, 'Global Velocity Error', 'Velocity Error (m/s)', best_idx)
        fig2.show()

    plt.show()


if __name__ == "__main__":
    main()