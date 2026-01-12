import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced
from lunwen1.chapter5.bayes_imm.adaptive_imm_lib import JilkovAdaptiveIMM
import lunwen1.chapter5.network.paper_plotting as pp


# ==========================================
# 1. 配置参数
# ==========================================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv'
DT = 1 / 30  # 30Hz 采样率
MEAS_NOISE_STD = 15 # 观测噪声标准差 (米)

def load_csv_data(filepath):
    """读取 CSV 并转换为 (9, N) 的状态矩阵"""
    try:
        df = pd.read_csv(filepath)
        # F-16数据包含9列: x,vx,ax, y,vy,ay, z,vz,az
        # 转置为 (9, N)
        state_matrix = df.values.T
        return state_matrix
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def run_filter(filter_obj, meas_pos, dt):
    num_steps = meas_pos.shape[1]
    # IMM 状态是 9 维 (x,vx,ax, y,vy,ay, z,vz,az)
    est_state = np.zeros((9, num_steps))
    model_probs = np.zeros((3, num_steps))

    # 初始状态记录
    est_state[:, 0] = filter_obj.x[0]
    model_probs[:, 0] = filter_obj.model_probs

    for i in range(1, num_steps):
        z = meas_pos[:, i]
        filter_obj.predict(dt)
        est, _ = filter_obj.update(z, dt)

        probs = filter_obj.model_probs  # 直接获取当前属性
        est_state[:, i] = est
        model_probs[:, i] = probs

    return est_state, model_probs

def main():
    # 1. 加载数据
    print(f"正在加载数据: {CSV_FILE_PATH} ...")
    true_state = load_csv_data(CSV_FILE_PATH)
    if true_state is None:
        return

    num_steps = true_state.shape[1]
    print(f"数据加载成功，共 {num_steps} 个采样点")

    # 2. 生成模拟观测值
    np.random.seed(42)  # 固定随机种子

    # --- 关键修正：适配 F-16 数据的 9维 索引 ---
    # 0:x, 1:vx, 2:ax, 3:y, 4:vy, 5:ay, 6:z, 7:vz, 8:az
    idx_pos = [0, 3, 6]
    idx_vel = [1, 4, 7]
    idx_acc = [2, 5, 8]

    true_pos = true_state[idx_pos, :]

    # 添加高斯白噪声
    meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    meas_pos = true_pos + meas_noise

    # 生成 R 矩阵 (观测噪声协方差)
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    # ==========================================
    # 【初始化】真值 + 微小扰动
    # ==========================================
    gt_init = true_state[:, 0]

    init_pos_err = 10.0
    init_vel_err = 5.0

    # 初始状态增加扰动
    init_noise = np.random.randn(9)
    init_noise[idx_pos] *= init_pos_err
    init_noise[idx_vel] *= init_vel_err
    init_noise[idx_acc] *= 1.0  # 加速度初始误差

    initial_state = gt_init + init_noise

    # 初始协方差 (9维对角阵)
    cov_diag = np.zeros(9)
    cov_diag[idx_pos] = init_pos_err ** 2
    cov_diag[idx_vel] = init_vel_err ** 2
    cov_diag[idx_acc] = 10.0
    initial_cov = np.diag(cov_diag)
    # ==========================================

    # 3. 定义转移概率矩阵 (Bo-IMM)
    # a, b, c, d, e, f = 0.99999, 0.000005, 0.000005, 0.99999, 0.000005, 0.000005 #todo 概率矩阵版
    a, b, c, d, e, f = 0.81388511, 0.18511489, 0.989, 0.01, 0.01, 0.01
    trans_pa = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [e, f, 1 - e - f]
    ])

    q, w, r, t, y, i = 0.989, 0.01, 0.11648423, 0.88251576, 0.86352624, 0.01
    trans_adp = np.array([
        [q, w, 1 - q - w],
        [r, t, 1 - r - t],
        [y, i, 1 - y - i]
    ])

    # 4. 初始化滤波器
    print("正在初始化滤波器...")
    imm_bo = IMMFilterEnhanced(trans_pa, initial_state, initial_cov, r_cov=r_cov)
    imm_adp = JilkovAdaptiveIMM(trans_adp, initial_state, initial_cov, r_cov=r_cov, window_len=45)

    # 5. 运行滤波
    print("正在运行 Bo-IMM ...")
    est_bo, probs_bo = run_filter(imm_bo, meas_pos, DT)

    print("正在运行 Adaptive-IMM ...")
    est_adp, probs_adp = run_filter(imm_adp, meas_pos, DT)

    # 6. 计算真实误差
    true_vel = true_state[idx_vel, :]
    true_acc = true_state[idx_acc, :]

    def calc_true_metrics(est):
        # 位置误差
        err_pos = est[idx_pos, :] - true_pos
        dist_err = np.sqrt(np.sum(err_pos ** 2, axis=0))
        # 速度误差
        err_vel = est[idx_vel, :] - true_vel
        vel_err = np.sqrt(np.sum(err_vel ** 2, axis=0))
        # 加速度误差
        err_acc = est[idx_acc, :] - true_acc
        acc_err = np.sqrt(np.sum(err_acc ** 2, axis=0))
        return dist_err, vel_err, acc_err

    dist_err_bo, vel_err_bo, acc_err_bo = calc_true_metrics(est_bo)
    dist_err_adp, vel_err_adp, acc_err_adp = calc_true_metrics(est_adp)

    EVAL_START_IDX = 80
    # 打印统计结果
    def print_stats(name, dist_err_p, dist_err_v, dist_err_a):
        rmse_p = np.sqrt(np.mean(dist_err_p[EVAL_START_IDX:] ** 2))
        rmse_v = np.sqrt(np.mean(dist_err_v[EVAL_START_IDX:] ** 2))
        rmse_a = np.sqrt(np.mean(dist_err_a[EVAL_START_IDX:] ** 2))
        print(f'{name:<10} | RMSE_p: {rmse_p:.4f} | RMSE_v: {rmse_v:.4f} | RMSE_a: {rmse_a:.4f}')

    print("-" * 100)
    print("真实误差统计 (Position, Velocity, Acceleration):")
    print_stats("Bo-IMM", dist_err_bo, vel_err_bo, acc_err_bo)
    print_stats("Adaptive-IMM", dist_err_adp, vel_err_adp, acc_err_adp)
    print("-" * 100)


    save_filename = 'res_data/adp_results.npz'
    print(f">>> 正在保存结果数据到: {save_filename} ...")

    # 构造时间轴
    t_axis = np.arange(num_steps) * DT

    np.savez(save_filename,
             t=t_axis,
             # 保存 Adaptive-IMM (Jilkov) 的误差
             err_adp_pos=dist_err_adp,
             err_adp_vel=vel_err_adp,
             )
    print(">>> 保存完成！")

    # ==========================================
    # 7. 绘图 (严格还原原始颜色和样式)
    # ==========================================
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


    PLOT_START = 80
    t_axis = np.arange(num_steps) * DT
    t_plot = t_axis[PLOT_START:]


    # --- 位置误差图 ---
    plt.figure(figsize=(10, 6))
    # 颜色还原：0.8-IMM(b), 0.6-IMM(m), 0.98-IMM(orange), Bo-IMM([0, 0.85, 0])
    plt.plot(t_plot, dist_err_bo[PLOT_START:], color=[0, 0.85, 0], label='Bo-IMM', linewidth=2.0)
    plt.plot(t_plot, dist_err_adp[PLOT_START:], 'm', label='ATP-IMM', linewidth=2.0)

    plt.title(f'位置误差对比 (Position RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- 速度误差图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, vel_err_bo[PLOT_START:], color=[0, 0.85, 0], label='Bo-IMM', linewidth=2.0)
    plt.plot(t_plot, vel_err_adp[PLOT_START:], 'm', label='ATP-IMM', linewidth=2.0)

    plt.title('速度误差对比 (Velocity RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s)')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- 加速度误差图 (原代码有此图，保留) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, acc_err_bo[PLOT_START:], color=[0, 0.85, 0], label='Bo-IMM', linewidth=2.0)
    plt.plot(t_plot, acc_err_adp[PLOT_START:], 'm', label='ATP-IMM', linewidth=2.0)

    plt.title('加速度误差对比 (Acceleration RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s^2)')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # ==========================================
    # 模型概率变化对比图 (2x2 布局)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.suptitle('IMM Model Probability Evolution Comparison', fontsize=16)

    plot_configs = [
        (probs_bo, "Bo-IMM", axes[0]),
        (probs_adp, "Adaptive-IMM", axes[1])
    ]

    # 还原原始标签和颜色
    model_names = ['Model 1 (CV)', 'Model 2 (CA)', 'Model 3 (CT)']

    styles = [
        # m=0: Model 1 (CV) - 蓝色
        # 设为宽线、半透明，作为"背景色"，放在最底层(zorder=1)
        {'color': 'tab:blue', 'alpha': 1.0, 'lw': 3.0, 'ls': '-', 'zorder': 1},

        # m=1: Model 2 (CA) - 橙色
        # 普通样式，中间层(zorder=2)
        {'color': 'tab:orange', 'alpha': 1.0, 'lw': 1.0, 'ls': '-', 'zorder': 2},

        # m=2: Model 3 (CT) - 绿色
        # 重点观察对象！设为不透明、实线、放在最顶层(zorder=3)，确保不被遮挡
        {'color': 'tab:green', 'alpha': 1.0, 'lw': 1.0, 'ls': '-', 'zorder': 3}
    ]

    for probs, title, ax in plot_configs:
        time_steps = np.arange(probs.shape[1]) * DT
        for m in range(3):
            ax.plot(time_steps, probs[m, :],
                    label=model_names[m],
                    color=styles[m]['color'],
                    linewidth=styles[m]['lw'],
                    linestyle=styles[m]['ls'],  # 关键：不同线型
                    alpha=styles[m]['alpha'],  # 关键：CV透，CT不透
                    zorder=styles[m]['zorder'])  # 关键：强制CT在最上面

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        # 把图例放到底部或外部，防止遮挡曲线
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 3D 轨迹图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 画真实轨迹
    ax.plot(true_state[idx_pos[0], :], true_state[idx_pos[1], :], true_state[idx_pos[2], :], 'k-', linewidth=1.5,
            label='真实轨迹')
    # 画 Bo-IMM 滤波轨迹
    ax.plot(est_bo[idx_pos[0], :], est_bo[idx_pos[1], :], est_bo[idx_pos[2], :], 'r-', linewidth=2, label='Bo-IMM 估计')

    ax.plot(est_adp[idx_pos[0], :], est_adp[idx_pos[1], :], est_adp[idx_pos[2], :], 'b--', linewidth=1.5,
            label='Adp-IMM 估计')

    # 画观测值 (稀疏显示)
    step_show = 10
    ax.scatter(meas_pos[0, ::step_show], meas_pos[1, ::step_show], meas_pos[2, ::step_show],
               s=1, c='gray', alpha=0.3, label='观测值')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('三维轨迹跟踪效果')
    ax.legend()

    print("正在寻找 Adaptive-IMM 表现持续优于 Bo-IMM 的最佳区间...")

    start_search = 100
    advantage_metric = dist_err_bo - dist_err_adp
    better_mask = advantage_metric > 0
    better_mask[:start_search] = False

    padded_mask = np.concatenate(([0], better_mask.astype(int), [0]))
    diffs = np.diff(padded_mask)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    best_segment = None
    max_score = -np.inf

    for s, e in zip(starts, ends):
        if e - s < 5: continue
        score = np.sum(advantage_metric[s:e])
        if score > max_score:
            max_score = score
            best_segment = (s, e)

    # ------------------------------------------------------------
    # 在这里调节你的“半径”！
    # ------------------------------------------------------------
    BEFORE_PLOT_RADIUS = -10  # <--- 就是这个，你想改成多少都行，像之前一样
    AFTER_PLOT_RADIUS = 35  # <--- 就是这个，你想改成多少都行，像之前一样

    if best_segment:
        s_idx, e_idx = best_segment
        print(f"  -> 找到最佳展示区间: Frames {s_idx} to {e_idx}")
        print(f"     在此区间内，Adp 累计比 Bo 减少了 {max_score:.2f} 米的误差")

        # 1. 在这段区间里，找到“最爽”的那一刻 (Peak Index)
        local_max_idx = s_idx + np.argmax(advantage_metric[s_idx:e_idx])
        print(f"     区间峰值时刻: Frame {local_max_idx} (Diff: {advantage_metric[local_max_idx]:.2f}m)")

        # 2. 以这个时刻为中心，向前后扩展固定半径 (回归你喜欢的逻辑)
        start_f = max(0, local_max_idx - BEFORE_PLOT_RADIUS)
        end_f = min(num_steps, local_max_idx + AFTER_PLOT_RADIUS)

    else:
        print("  -> 未找到 Adp 显著优胜区间，退回全局最大差异点...")
        best_idx = np.argmax(advantage_metric[start_search:]) + start_search
        start_f = max(0, best_idx - BEFORE_PLOT_RADIUS)
        end_f = min(num_steps, best_idx + AFTER_PLOT_RADIUS)

    print(f"  -> 最终绘图窗口: Frame {start_f} to {end_f} (Radius: {AFTER_PLOT_RADIUS + BEFORE_PLOT_RADIUS})")

    est_dict = {
        'Adaptive-IMM': {
            'data': est_adp[[0, 3, 6], :].T,
            'color': 'm', 'style': '-', 'width': 2.0, 'alpha': 0.9
        },
        'Bo-IMM': {
            'data': est_bo[[0, 3, 6], :].T,
            'color': [0, 0.8, 0], 'style': '--', 'width': 2.5, 'alpha': 0.6
        }
    }
    true_pos_T = true_state[[0, 3, 6], :].T
    pp.plot_3d_zoom_multi(true_pos_T, est_dict, start_f, end_f)

    # =================================================================
    # [新增功能] 局部细节对比图 (仿 compare_bo_adp_bayes.py 风格)
    # 目标: 寻找并绘制 Adaptive-IMM < Bo-IMM (即 Adaptive 更优) 的窗口
    # =================================================================
    print("-" * 50)
    print(">>> 正在生成局部细节对比图 (Pos/Vel/Acc)...")

    # 1. 样式配置 (复刻 compare 脚本风格)
    # 逻辑: Adaptive-IMM (Best, Green) vs Bo-IMM (Worst, Blue)
    LOCAL_STYLES = {
        'Adaptive-IMM': {
            # 对应 Best: 深绿, 实线, 加号
            'c': 'm', 'ls': '-', 'mk': '+', 'ms': 9, 'lw': 2.0, 'alpha': 1.0, 'zorder': 10,
            'label': 'ATP-IMM'
        },
        'Bo-IMM': {
            # 对应 Worst: 蓝色, 虚线, 圆圈
            'c': [0, 0.8, 0], 'ls': '--', 'mk': 'o', 'ms': 6, 'lw': 1.5, 'alpha': 0.7, 'zorder': 5,
            'label': 'Bo-IMM'
        }
    }

    # 2. 自动搜索最佳展示窗口
    # 目标: 寻找 Adaptive-IMM 误差显著小于 Bo-IMM 的区域
    ZOOM_WIN_SIZE = 100  # 窗口大小

    candidates = []

    # 从稳定后开始搜索 (跳过前100帧)
    search_start = 100
    search_end = num_steps - ZOOM_WIN_SIZE

    print(f"  -> 正在搜索 Adaptive-IMM 优势窗口 (Window Size: {ZOOM_WIN_SIZE})...")

    for k in range(search_start, search_end):
        # 获取窗口数据
        seg_bo = dist_err_bo[k: k + ZOOM_WIN_SIZE]
        seg_adp = dist_err_adp[k: k + ZOOM_WIN_SIZE]

        m_bo = np.mean(seg_bo)
        m_adp = np.mean(seg_adp)

        # 计算差距 (Bo - Adp)，正值代表 Adp 更准
        gap = m_bo - m_adp

        # 一致性检查：窗口内有多少比例 Adp 是优于 Bo 的
        consistency = np.sum(seg_adp < seg_bo) / ZOOM_WIN_SIZE

        # 评分: 差距 * 一致性 (优先选差距大且表现稳定的区域)
        score = gap * (consistency ** 2)

        # 筛选条件: 平均误差更小，且至少60%的时间步更优
        if m_adp < m_bo and consistency > 0.6:
            candidates.append((k, score))

    # 决策
    best_win_idx = -1

    if len(candidates) > 0:
        # 找分数最高的 (差距最大)
        best_win_idx, best_score = max(candidates, key=lambda x: x[1])
        print(f"  [成功] 找到 Adaptive-IMM 优势窗口: Frame {best_win_idx}")
        print(f"     -> 平均误差优势 (Gap): {best_score:.4f}m")
    else:
        # 备选: 如果没找到显著优势，找个差距最小的(或者随便最后一段)
        best_win_idx = search_end - 1
        print("  [提示] 未找到显著优势窗口，默认显示最后一段。")

    # 3. 准备绘图数据
    slice_idx = slice(best_win_idx, best_win_idx + ZOOM_WIN_SIZE)
    x_local = np.arange(ZOOM_WIN_SIZE)

    plot_data_map = {
        'Position Error (m)': {
            'Adaptive-IMM': dist_err_adp[slice_idx],
            'Bo-IMM': dist_err_bo[slice_idx]
        },
        'Velocity Error (m/s)': {
            'Adaptive-IMM': vel_err_adp[slice_idx],
            'Bo-IMM': vel_err_bo[slice_idx]
        },
        'Acceleration Error (m/s^2)': {  # 避免中文 SimHei 报错
            'Adaptive-IMM': acc_err_adp[slice_idx],
            'Bo-IMM': acc_err_bo[slice_idx]
        }
    }

    # 4. 绘制 3x1 子图
    fig_zoom, axes_zoom = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_zoom.suptitle(f'Local Detail Comparison: Frames {best_win_idx}-{best_win_idx + ZOOM_WIN_SIZE} (Adp < Bo)',
                      fontsize=16)

    metric_names = ['Position Error (m)', 'Velocity Error (m/s)', 'Acceleration Error (m/s^2)']
    # 绘图顺序: 先画 Bo(差)，后画 Adp(好)，保证好的线在上面
    draw_order = ['Bo-IMM', 'Adaptive-IMM']

    for i, metric in enumerate(metric_names):
        ax = axes_zoom[i]
        data_group = plot_data_map[metric]

        for model_name in draw_order:
            y_data = data_group[model_name]
            style = LOCAL_STYLES[model_name]

            ax.plot(x_local, y_data,
                    color=style['c'],
                    linestyle=style['ls'],
                    marker=style['mk'],
                    markersize=style['ms'],
                    linewidth=style['lw'],
                    alpha=style['alpha'],
                    zorder=style['zorder'],
                    label=style['label'],
                    markevery=5)  # 标记间隔，防止拥挤

        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        # 仅第一幅图显示图例 (右上角，垂直排列)
        if i == 0:
            ax.legend(loc='upper right', ncol=1, fontsize=10,
                      framealpha=0.95, shadow=True, borderpad=0.6)

        # 动态 Y 轴范围 (留出头部空间给图例)
        all_vals = np.concatenate(list(data_group.values()))
        ax.set_ylim(0, np.max(all_vals) * 1.45)

    axes_zoom[-1].set_xlabel(f'Time Step (k)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

if __name__ == "__main__":
    main()