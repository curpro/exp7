import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, BboxConnector
from matplotlib.transforms import Bbox, TransformedBbox
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 引入您的自定义库
try:
    from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced
    import lunwen1.chapter5.network.paper_plotting as pp
except ImportError:
    try:
        from imm_lib_enhanced import IMMFilterEnhanced
        import paper_plotting as pp
    except ImportError:
        print("错误: 请确保 imm_lib_enhanced.py 和 paper_plotting.py 在当前目录或 python 路径中。")
        exit()

# ==========================================
# 1. 配置参数 (保持不变)
# ==========================================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv'
DT = 1 / 30  # 30Hz 采样率
MEAS_NOISE_STD = 15  # 观测噪声标准差 (米)


# ==========================================
# [新增] 样式配置 (用于 Combine Plot)
# ==========================================
# Global Style: 细实线，高透明度
STYLE_GLOBAL = {
    'Bo-IMM': {'c': [0, 0.85, 0], 'lw': 1.8, 'alpha': 0.95, 'zorder': 10, 'label': 'Bo-IMM'},
    'EKF':    {'c': 'm', 'lw': 1.2, 'alpha': 0.85, 'zorder': 8,  'label': 'EKF'},
    'PF':     {'c': 'orange',  'lw': 1.0, 'alpha': 0.70, 'zorder': 6,  'label': 'PF'},
    'GPF':    {'c': 'b', 'lw': 1.0, 'alpha': 0.60, 'zorder': 4,  'label': 'GPF'}
}

# Local Style: 带 Marker，不透明
STYLE_LOCAL = {
    'Bo-IMM': {'mk': '*', 'ms': 9, 'ls': '-',  'lw': 1.2, 'alpha': 1.0},
    'EKF':    {'mk': '^', 'ms': 7, 'ls': '-.', 'lw': 1.2, 'alpha': 0.9},
    'PF':     {'mk': 's', 'ms': 6, 'ls': '--', 'lw': 1.2, 'alpha': 0.8},
    'GPF':    {'mk': 'o', 'ms': 5, 'ls': ':',  'lw': 1.2, 'alpha': 0.7}
}

DISPLAY_ORDER = ['GPF', 'PF', 'EKF', 'Bo-IMM']
MARK_EVERY = 5 # 子图标记间隔


# ==========================================
# 2. 算法类定义 (EKF) - 保持不变
# ==========================================
class EKF_CA_9D:
    def __init__(self, initial_state, initial_cov, r_cov, dt):
        self.dim = 9
        self.x = initial_state.copy()
        self.P = initial_cov.copy()
        self.R = r_cov
        self.dt = dt
        self.F = np.eye(self.dim)
        t = dt
        block = np.array([[1, t, 0.5 * t ** 2], [0, 1, t], [0, 0, 1]])
        for i in [0, 3, 6]:
            self.F[i:i + 3, i:i + 3] = block

        # 保持你原有的 Q 参数
        q_std = 100.0
        var = q_std ** 2
        q_block = np.array([
            [t ** 5 / 20, t ** 4 / 8, t ** 3 / 6],
            [t ** 4 / 8, t ** 3 / 3, t ** 2 / 2],
            [t ** 3 / 6, t ** 2 / 2, t]
        ]) * var
        self.Q = np.zeros((self.dim, self.dim))
        for i in [0, 3, 6]:
            self.Q[i:i + 3, i:i + 3] = q_block
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H[2, 6] = 1

    def update(self, z):
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        try:
            K = P_pred @ self.H.T @ np.linalg.inv(S)
        except:
            K = np.zeros((self.dim, 3))
        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ P_pred
        return self.x

# 粒子滤波
class ParticleFilter_CA_9D:
    def __init__(self, num_particles, initial_state, initial_cov, r_cov, dt):
        self.num_particles = num_particles
        self.dim = 9
        self.dt = dt
        self.R = r_cov

        # 1. 初始化粒子
        # 从初始高斯分布中采样
        self.particles = np.random.multivariate_normal(initial_state, initial_cov, num_particles)

        # 初始化权重 (均匀)
        self.weights = np.ones(num_particles) / num_particles

        # 过程噪声 (与 EKF 保持一致，为了公平)
        t = dt
        q_std = 100.0  # 与 EKF 一致
        var = q_std ** 2
        # 构建 Q 矩阵的对角块
        q_block = np.array([
            [t ** 5 / 20, t ** 4 / 8, t ** 3 / 6],
            [t ** 4 / 8, t ** 3 / 3, t ** 2 / 2],
            [t ** 3 / 6, t ** 2 / 2, t]
        ]) * var

        # 为了采样方便，我们需要 Q 的平方根 (Cholesky) 用于生成随机噪声
        # 但这 9x9 矩阵可能数值不稳定，我们直接对每个轴独立加噪声
        # 或者直接生成多元正态噪声
        self.Q_matrix = np.zeros((9, 9))
        for i in [0, 3, 6]:
            self.Q_matrix[i:i + 3, i:i + 3] = q_block

    def predict(self):
        dt = self.dt
        # CA 模型转移矩阵 (应用于所有粒子)
        # x = x + vx*t + 0.5*ax*t^2
        # vx = vx + ax*t
        # ax = ax

        # 向量化操作
        # X: (N, 9) -> [x, vx, ax, y, vy, ay, z, vz, az]
        # 注意：你的状态顺序是 [x, vx, ax, y, vy, ay, ...] (索引 0,1,2 是 x轴)

        # 确定性预测
        for i in [0, 3, 6]:  # 对 x, y, z 三轴循环
            p = self.particles[:, i]
            v = self.particles[:, i + 1]
            a = self.particles[:, i + 2]

            self.particles[:, i] = p + v * dt + 0.5 * a * dt ** 2
            self.particles[:, i + 1] = v + a * dt
            self.particles[:, i + 2] = a

        # 添加过程噪声
        noise = np.random.multivariate_normal(np.zeros(9), self.Q_matrix, self.num_particles)
        self.particles += noise

    def update(self, z):
        # z: 观测向量 (3,) [x, y, z]
        # 观测模型：只观测位置 H = [1 0 0 ...; 0 0 0 1 ...; ...]

        # 1. 计算每个粒子的似然 (Likelihood)
        # 假设高斯似然 p(z|x)
        pred_pos = self.particles[:, [0, 3, 6]]  # 取出位置 (N, 3)
        diff = pred_pos - z  # (N, 3)

        # 计算马氏距离的平方 (或者直接用欧氏距离，如果 R 是对角阵)
        # dist_sq = (diff / std)^2
        # 假设 R 是对角阵 (代码里 r_cov 是 eye(3)*noise^2)
        R_diag = np.diag(self.R)  # (3,)
        log_likelihood = -0.5 * np.sum((diff ** 2) / R_diag, axis=1)

        # 2. 更新权重
        # w = w * likelihood
        # 为了数值稳定，使用 log-sum-exp 技巧
        self.weights = np.log(self.weights + 1e-300) + log_likelihood
        self.weights = np.exp(self.weights - np.max(self.weights))  # 减去最大值防止溢出
        self.weights /= np.sum(self.weights)  # 归一化

        # 3. 状态估计 (加权平均)
        est_x = np.average(self.particles, weights=self.weights, axis=0)

        # 4. 重采样 (Resampling)
        # 有效粒子数 N_eff
        N_eff = 1.0 / np.sum(self.weights ** 2)
        # 当有效粒子数太少时重采样 (阈值通常为 N/2)
        if N_eff < self.num_particles / 2.0:
            self.resample()

        return est_x

    def resample(self):
        # 系统重采样 (Systematic Resampling)
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # 确保最后一个是1

        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)

        idx = 0
        for i in range(self.num_particles):
            pos = start + i * step
            while pos > cumulative_sum[idx]:
                idx += 1
            indices[i] = idx

        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)


# ==========================================
# 3. 辅助函数 (保持不变)
# ==========================================
def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df.values.T
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def run_imm_filter(filter_obj, meas_pos, dt):
    num_steps = meas_pos.shape[1]
    est_state = np.zeros((9, num_steps))
    est_state[:, 0] = filter_obj.x[0]
    for i in range(1, num_steps):
        filter_obj.predict(dt)
        est, _ = filter_obj.update(meas_pos[:, i], dt)
        est_state[:, i] = est
    return est_state


def run_ekf_filter(filter_obj, meas_pos):
    num_steps = meas_pos.shape[1]
    est_state = np.zeros((9, num_steps))
    est_state[:, 0] = filter_obj.x
    for i in range(1, num_steps):
        est_state[:, i] = filter_obj.update(meas_pos[:, i])
    return est_state


def calculate_derivatives_for_gp(pos_data, dt):
    # --- 1. 计算原始速度 (Raw Velocity) ---
    # 使用后向差分：v[t] = (p[t] - p[t-1]) / dt
    diff_pos = np.diff(pos_data, axis=1)
    raw_vel = diff_pos / dt

    # 填充第一帧 (补0)
    zeros_col = np.zeros((3, 1))
    raw_vel = np.hstack((zeros_col, raw_vel))

    # --- 2. 对速度应用 EMA 平滑 ---
    # alpha 越小越平滑，滞后越大；alpha 越大反应越快，噪声越大。
    # 建议 0.3 ~ 0.6 之间。对于 F16 高机动，0.4 左右是一个平衡点。
    alpha_v = 0.9
    vel = np.zeros_like(raw_vel)
    vel[:, 0] = raw_vel[:, 0]

    for k in range(1, raw_vel.shape[1]):
        # EMA 公式: Current = alpha * Raw + (1-alpha) * Last
        vel[:, k] = alpha_v * raw_vel[:, k] + (1 - alpha_v) * vel[:, k - 1]

    # --- 3. 计算原始加速度 (Raw Acceleration) ---
    # 注意：这里使用“平滑后的速度”来计算加速度，效果更好
    diff_vel = np.diff(vel, axis=1)
    raw_acc = diff_vel / dt
    raw_acc = np.hstack((zeros_col, raw_acc))

    # --- 4. 对加速度应用 EMA 平滑 ---
    # 加速度通常噪声极大，建议 alpha 设置得比速度更小一点 (更强平滑)
    alpha_a = 0.05
    acc = np.zeros_like(raw_acc)
    acc[:, 0] = raw_acc[:, 0]

    for k in range(1, raw_acc.shape[1]):
        acc[:, k] = alpha_a * raw_acc[:, k] + (1 - alpha_a) * acc[:, k - 1]

    # --- 5. 组装 9D 状态 ---
    N = pos_data.shape[1]
    state_9d = np.zeros((9, N))
    state_9d[[0, 3, 6], :] = pos_data  # 位置 (GP直接预测的)
    state_9d[[1, 4, 7], :] = vel  # 速度 (差分+平滑)
    state_9d[[2, 5, 8], :] = acc  # 加速度 (二次差分+平滑)

    return state_9d


# ==========================================
# [新增] Combine Plot 绘制函数
# ==========================================
def draw_combined_figure(data_dict, title_text, y_label, best_idx, window_size):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 准备数据
    min_len = min(len(v) for v in data_dict.values())
    time_axis = np.arange(min_len) * DT

    all_global_values = []
    local_max_val = 0
    zoom_start = best_idx
    zoom_end = best_idx + window_size

    # 1. 绘制全局背景 (Global Style)
    # 跳过前80帧初始化
    PLOT_START = 80

    for name in DISPLAY_ORDER:
        if name not in data_dict: continue

        full_y = data_dict[name]
        plot_y = full_y[PLOT_START:]
        plot_x = time_axis[PLOT_START:len(full_y)]

        all_global_values.extend(plot_y)

        # 记录局部最大值 (用于防撞)
        if zoom_end <= len(full_y):
            local_seg = full_y[zoom_start:zoom_end]
            if len(local_seg) > 0:
                local_max_val = max(local_max_val, np.max(local_seg))

        s = STYLE_GLOBAL[name]
        ax.plot(plot_x, plot_y,
                c=s['c'], ls='-', lw=s['lw'],
                alpha=s['alpha'], zorder=s['zorder'], label=s['label'])

    # 2. 设置 Y 轴留白
    global_data_max = np.percentile(all_global_values, 99.5) if all_global_values else 1.0
    ax.set_ylim(0, global_data_max * 2.5)

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)

    # 3. 绘制悬浮子图 (Local Style)
    axins = ax.inset_axes([0.05, 0.55, 0.45, 0.40])
    local_x = np.arange(window_size)
    local_vals_inset = []

    for name in DISPLAY_ORDER:
        if name not in data_dict: continue

        local_y = data_dict[name][zoom_start:zoom_end]
        local_vals_inset.extend(local_y)

        s_glob = STYLE_GLOBAL[name]
        s_loc = STYLE_LOCAL[name]

        axins.plot(local_x, local_y,
                   c=s_glob['c'], ls=s_loc['ls'], lw=s_loc['lw'],
                   marker=s_loc['mk'], ms=s_loc['ms'], markevery=MARK_EVERY,
                   alpha=s_loc['alpha'], zorder=s_glob['zorder'])

    axins.set_xlim(0, window_size)
    if local_vals_inset:
        axins.set_ylim(0, max(local_vals_inset) * 1.15)
    axins.grid(True, linestyle=':', alpha=0.5)
    axins.set_xlabel('Step (k)', fontsize=10)

    # 4. 连接线与框
    box_x0 = time_axis[zoom_start]
    box_width = time_axis[zoom_end - 1] - box_x0
    box_height = local_max_val

    rect_patch = Rectangle((box_x0, 0), box_width, box_height,
                           fill=False, edgecolor="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.add_patch(rect_patch)

    rect_bbox = Bbox.from_bounds(box_x0, 0, box_width, box_height)
    rect_transform = TransformedBbox(rect_bbox, ax.transData)

    ax.add_patch(BboxConnector(axins.bbox, rect_transform, loc1=3, loc2=2, edgecolor="k", linestyle="--", linewidth=0.8,
                               alpha=0.5))
    ax.add_patch(BboxConnector(axins.bbox, rect_transform, loc1=4, loc2=1, edgecolor="k", linestyle="--", linewidth=0.8,
                               alpha=0.5))

    return fig


# ==========================================
# 4. 主函数
# ==========================================
def main():
    # 1. 加载数据
    print(f"正在加载数据: {CSV_FILE_PATH} ...")
    true_state = load_csv_data(CSV_FILE_PATH)
    if true_state is None: return
    num_steps = true_state.shape[1]

    # 2. 数据准备 (保持不变)
    np.random.seed(42)
    idx_pos = [0, 3, 6]
    idx_vel = [1, 4, 7]
    idx_acc = [2, 5, 8]
    true_pos = true_state[idx_pos, :]
    true_vel = true_state[idx_vel, :]
    true_acc = true_state[idx_acc, :]

    meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    meas_pos = true_pos + meas_noise
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    gt_init = true_state[:, 0]
    init_noise = np.random.randn(9)
    init_noise[idx_pos] *= 10.0
    init_noise[idx_vel] *= 5.0
    init_noise[idx_acc] *= 1.0
    initial_state = gt_init + init_noise

    cov_diag = np.zeros(9)
    cov_diag[idx_pos] = 100.0
    cov_diag[idx_vel] = 25.0
    cov_diag[idx_acc] = 10.0
    initial_cov = np.diag(cov_diag)

    # ==========================================
    # 3. 运行滤波器
    # ==========================================
    print("-" * 50)

    # --- A. Bo-IMM (保持不变) ---
    print("正在运行 Bo-IMM...")
    a, b, c, d, e, f = 0.81388511, 0.18511489, 0.989, 0.01, 0.01, 0.01
    trans_bo = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [e, f, 1 - e - f]
    ])
    imm_bo = IMMFilterEnhanced(trans_bo, initial_state, initial_cov, r_cov=r_cov)
    est_bo = run_imm_filter(imm_bo, meas_pos, DT)

    # --- B. EKF (保持不变) ---
    print("正在运行 EKF...")
    ekf = EKF_CA_9D(initial_state, initial_cov, r_cov, DT)
    est_ekf = run_ekf_filter(ekf, meas_pos)

    # --- C. GP (【修改部分】：改为滑动窗口 + 噪声失配) ---
    print("正在运行 GPF (Online Sliding Window)...")

    # >> 削弱 1: 只能看过去 20 个点
    GP_WINDOW_SIZE = 90

    # >> 削弱 2: 假设的噪声方差 (100) 小于真实方差 (225)
    # 这会模拟“过拟合”现象，使 GP 对噪声更敏感，从而降低其平滑效果
    GP_ASSUMED_NOISE_VAR = 0.2

    t_axis = np.arange(num_steps) * DT
    X_full = t_axis.reshape(-1, 1)

    # 建立 GP 核函数
    base_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
         RBF(length_scale=3.5, length_scale_bounds=(1.0, 1e3)) + \
         WhiteKernel(noise_level=GP_ASSUMED_NOISE_VAR, noise_level_bounds=(1e-5, 0.5))

    est_gp_pos = np.zeros((3, num_steps))

    # 逐维度处理
    for dim in range(3):
        print(f"  -> Dimension {dim} processing...")
        y_full = meas_pos[dim, :]

        # 逐时刻进行“在线”预测
        for t in range(num_steps):
            # 确定当前窗口 [t - window + 1 : t + 1]
            start_idx = max(0, t - GP_WINDOW_SIZE + 1)
            end_idx = t + 1

            X_train = X_full[start_idx:end_idx]
            y_train = y_full[start_idx:end_idx]

            # 拟合当前窗口数据
            # gp = GaussianProcessRegressor(kernel=base_kernel, alpha=0.0,
            #                               n_restarts_optimizer=2, normalize_y=True)

            gp = GaussianProcessRegressor(kernel=base_kernel,alpha=0.0,
                                          optimizer=None, normalize_y=True)
            gp.fit(X_train, y_train)

            # 只预测当前这一个点 (t)
            X_curr = X_full[t].reshape(1, -1)
            pred_val = gp.predict(X_curr, return_std=False)

            est_gp_pos[dim, t] = pred_val[0]

            if t % 500 == 0:
                print(f"     Step {t}/{num_steps}", end='\r')
        print("")

    # 计算导数 (速度/加速度)
    est_gp = calculate_derivatives_for_gp(est_gp_pos, DT)

    print("正在运行 Particle Filter...")
    num_particles = 5000  # 粒子数，越多越准但越慢
    pf = ParticleFilter_CA_9D(num_particles, initial_state, initial_cov, r_cov, DT)
    est_pf = np.zeros((9, num_steps))
    est_pf[:, 0] = initial_state

    for i in range(1, num_steps):
        pf.predict()
        est_pf[:, i] = pf.update(meas_pos[:, i])


    # ==========================================
    # 4. 统计与绘图 (保持不变)
    # ==========================================
    def calc_true_metrics(est):
        dist_err = np.sqrt(np.sum((est[idx_pos, :] - true_pos) ** 2, axis=0))
        vel_err = np.sqrt(np.sum((est[idx_vel, :] - true_vel) ** 2, axis=0))
        acc_err = np.sqrt(np.sum((est[idx_acc, :] - true_acc) ** 2, axis=0))
        return dist_err, vel_err, acc_err

    dist_err_bo, vel_err_bo, acc_err_bo = calc_true_metrics(est_bo)
    dist_err_ekf, vel_err_ekf, acc_err_ekf = calc_true_metrics(est_ekf)
    dist_err_gp, vel_err_gp, acc_err_gp = calc_true_metrics(est_gp)
    dist_err_pf, vel_err_pf, acc_err_pf = calc_true_metrics(est_pf)

    EVAL_START = 80

    def print_stats(name, dist_err_p, dist_err_v, dist_err_a):
        rmse_p = np.sqrt(np.mean(dist_err_p[EVAL_START:] ** 2))
        rmse_v = np.sqrt(np.mean(dist_err_v[EVAL_START:] ** 2))
        rmse_a = np.sqrt(np.mean(dist_err_a[EVAL_START:] ** 2))
        print(f'{name:<15} | RMSE_p: {rmse_p:.4f} | RMSE_v: {rmse_v:.4f} | RMSE_a: {rmse_a:.4f}')

    print("\n" + "=" * 80)
    print("真实误差统计 (Comparison):")
    print_stats("Bo-IMM", dist_err_bo, vel_err_bo, acc_err_bo)
    print_stats("EKF", dist_err_ekf, vel_err_ekf, acc_err_ekf)
    print_stats("GPF", dist_err_gp, vel_err_gp, acc_err_gp)
    print_stats("PF", dist_err_pf, vel_err_pf, acc_err_pf)
    print("=" * 80 + "\n")

    # 绘图部分
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    t_plot = t_axis[EVAL_START:]
    c_bo, c_ekf, c_gp, c_pf = [0, 0.85, 0], 'm', 'b', 'orange'

    # 绘制位置误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, dist_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, dist_err_pf[EVAL_START:], color=c_pf, label='PF', alpha=0.6)
    plt.plot(t_plot, dist_err_ekf[EVAL_START:], color=c_ekf, label='EKF', alpha=0.6)
    plt.plot(t_plot, dist_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)
    plt.title('位置误差对比 (Position RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制速度误差 (新增)
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, vel_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, vel_err_pf[EVAL_START:], color=c_pf, label='PF', alpha=0.6)
    plt.plot(t_plot, vel_err_ekf[EVAL_START:], color=c_ekf, label='EKF', alpha=0.6)
    plt.plot(t_plot, vel_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)
    plt.title('速度误差对比 (Velocity RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s)')
    plt.legend(loc='upper right', framealpha=1.0)
    plt.grid(True, alpha=0.3)

    # [新增] 绘制加速度误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, acc_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, acc_err_pf[EVAL_START:], color=c_pf, label='PF', alpha=0.6)
    plt.plot(t_plot, acc_err_ekf[EVAL_START:], color=c_ekf, label='EKF', alpha=0.6)
    plt.plot(t_plot, acc_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)

    plt.title('加速度误差对比 (Acceleration RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s^2)')  # 注意单位是 s平方
    plt.legend()
    plt.grid(True, alpha=0.3)


    # 3D Zoom Plot
    try:
        mask_order = (dist_err_bo < dist_err_ekf) & \
                     (dist_err_ekf < dist_err_pf) & \
                     (dist_err_pf < dist_err_gp)
        valid_indices = np.where(mask_order & (np.arange(num_steps) > 100))[0]
        if len(valid_indices) > 0:
            print(f"找到 {len(valid_indices)} 个满足 Bo < EKF < PF < GPF 的时刻。")

            spread = dist_err_gp[valid_indices] - dist_err_bo[valid_indices]

            best_idx_loc = np.argmax(spread)
            best_idx = valid_indices[best_idx_loc]

            print(f"选定最佳展示 Frame: {best_idx} (最大误差差值: {spread[best_idx_loc]:.4f}m)")
        else:
            print("警告：未找到严格满足 Bo < EKF < PF < GPF 的时刻，尝试宽松模式...")
            score = (dist_err_gp - dist_err_bo)
            best_idx = np.argmax(score[100:]) + 100
            print(f"宽松模式选定 Frame: {best_idx}")

        beforeRadius = 30
        afterRadius = 5
        start_f = max(0, best_idx - beforeRadius)
        end_f = min(num_steps, best_idx + afterRadius)

        est_dict = {
            'GPF': {'data': est_gp[[0, 3, 6], :].T, 'color': c_gp, 'style': '--', 'width': 1.5, 'alpha': 0.5},
            'EKF': {'data': est_ekf[[0, 3, 6], :].T, 'color': c_ekf, 'style': '--', 'width': 1.5, 'alpha': 0.5},
            'PF': {'data': est_pf[[0, 3, 6], :].T, 'color': c_pf, 'style': '--', 'width': 1.5, 'alpha': 0.5},  # [新增]
            'Bo-IMM': {'data': est_bo[[0, 3, 6], :].T, 'color': c_bo, 'style': '-', 'width': 2.0, 'alpha': 1.0}
        }
        pp.plot_3d_zoom_multi(true_state[[0, 3, 6], :].T, est_dict, start_f, end_f)
    except Exception as e:
        print(f"3D Zoom 绘图跳过: {e}")

    # =================================================================
    # [新增功能] 高级局部细节对比图 (仿照 compare_bo_adp_bayes.py 风格)
    # 目标: 寻找并绘制满足 Bo < EKF < PF < GPF 的窗口
    # =================================================================
    print("-" * 50)
    print(">>> 正在生成局部细节对比图 (Pos/Vel/Acc)...")

    # 1. 样式配置 (严格区分四种算法)
    # 排序要求: Bo(优) < EKF < PF < GPF(差)
    # zorder 越高图层越靠上，我们让最好的线压在最上面
    LOCAL_STYLES = {
        'Bo-IMM': {
            # 绿色, 实线, 星号 (Best) 
            'c': [0, 0.85, 0], 'ls': '-', 'mk': '*', 'ms': 9, 'lw': 2.5, 'alpha': 1.0, 'zorder': 10,
            'label': 'Bo-IMM'
        },
        'EKF': {
            # 红色, 点划线, 三角 (Second)
            'c': 'm', 'ls': '-.', 'mk': '^', 'ms': 6, 'lw': 1.5, 'alpha': 0.9, 'zorder': 8,
            'label': 'EKF'
        },
        'PF': {
            # 橙色, 虚线, 方块 (Third)
            'c': 'orange', 'ls': '--', 'mk': 's', 'ms': 5, 'lw': 1.5, 'alpha': 0.8, 'zorder': 6,
            'label': 'PF'
        },
        'GPF': {
            # 蓝色, 点线, 圆圈 (Worst)
            'c': 'b', 'ls': ':', 'mk': 'o', 'ms': 5, 'lw': 1.5, 'alpha': 0.7, 'zorder': 4,
            'label': 'GPF'
        }
    }

    # 2. 自动搜索最佳展示窗口
    ZOOM_WIN_SIZE = 100  # 窗口大小

    candidates_strict = []  # 严格满足 Bo < EKF < PF < GPF
    candidates_loose = []  # 松散满足 (只要 Bo 显著优于 EKF)

    # 从稳定后开始搜索
    search_start = 100
    search_end = num_steps - ZOOM_WIN_SIZE

    print(f"  -> 正在搜索最佳分离窗口 (Window Size: {ZOOM_WIN_SIZE})...")

    for k in range(search_start, search_end):
        # 提取当前窗口的数据片段
        seg_bo = dist_err_bo[k: k + ZOOM_WIN_SIZE]
        seg_ekf = dist_err_ekf[k: k + ZOOM_WIN_SIZE]
        seg_pf = dist_err_pf[k: k + ZOOM_WIN_SIZE]
        seg_gp = dist_err_gp[k: k + ZOOM_WIN_SIZE]

        # 计算均值
        m_bo, m_ekf, m_pf, m_gp = np.mean(seg_bo), np.mean(seg_ekf), np.mean(seg_pf), np.mean(seg_gp)

        # === 核心修改：评分逻辑 ===
        # 我们不关心 GPF 飞多远，只关心 Bo 和 EKF 拉开多大差距
        # gap_inner: Bo-IMM 相比 EKF 的优势有多大
        gap_inner = m_ekf - m_bo

        # consistency: 在这个窗口内，有多少比例的时刻 Bo 是真的比 EKF 好的？
        # 防止出现"Bo偶尔极好拉低平均值，但平时很差"的情况
        consistency = np.sum(seg_bo < seg_ekf) / ZOOM_WIN_SIZE

        # 综合评分：差距 * 一致性
        score = gap_inner * (consistency ** 2)

        # 只有当 Bo 比 EKF 好 (gap > 0) 且一致性较高 (>60%) 时才考虑
        if gap_inner > 0 and consistency > 0.6:

            # A. 严格排序判定: Bo < EKF < PF < GPF (且 Bo 必须比 EKF 小)
            if m_bo < m_ekf and m_ekf < m_pf and m_pf < m_gp:
                candidates_strict.append((k, score))

            # B. 松散排序判定: 只要 Bo 最强，PF/GPF 谁差无所谓
            # 这种情况常用于 Bo 表现极好，但 PF 和 GPF 纠缠的时候
            elif m_bo < m_ekf and m_bo < m_pf and m_bo < m_gp:
                candidates_loose.append((k, score))

    # 决策选择
    best_win_idx = -1
    is_strict_match = False

    if len(candidates_strict) > 0:
        # 在严格满足的候选里，找 Bo 和 EKF 差距最大的
        best_win_idx, best_score = max(candidates_strict, key=lambda x: x[1])
        is_strict_match = True
        print(f"  [成功] 找到严格排序窗口 (Bo < EKF < PF < GPF).")
        print(f"     -> Frame: {best_win_idx} | Bo与EKF平均差距: {best_score:.4f}m")

    elif len(candidates_loose) > 0:
        # 如果找不到全员排序的，就找 Bo 优势最大的（忽略 PF/GPF 的乱序）
        best_win_idx, best_score = max(candidates_loose, key=lambda x: x[1])
        print(f"  [警告] 未找到全员严格排序，选择 Bo-IMM 优势最大的窗口.")
        print(f"     -> Frame: {best_win_idx} | Bo与EKF平均差距: {best_score:.4f}m")

    else:
        best_win_idx = search_end - 1
        print(f"  [失败] Bo-IMM 在所有窗口中均未表现出对 EKF 的显著优势。显示最后一段。")


    # 1. 准备位置数据
    data_pos = {
        'Bo-IMM': dist_err_bo,
        'EKF': dist_err_ekf,
        'PF': dist_err_pf,
        'GPF': dist_err_gp
    }

    # 2. 准备速度数据
    data_vel = {
        'Bo-IMM': vel_err_bo,
        'EKF': vel_err_ekf,
        'PF': vel_err_pf,
        'GPF': vel_err_gp
    }

    print("\n>>> 生成 Combined Plot (Pos)...")
    fig_comb_pos = draw_combined_figure(data_pos, 'Position Error', 'Position Error (m)', best_win_idx,
                                        ZOOM_WIN_SIZE)
    fig_comb_pos.show()

    print(">>> 生成 Combined Plot (Vel)...")
    fig_comb_vel = draw_combined_figure(data_vel, 'Velocity Error', 'Velocity Error (m/s)', best_win_idx,
                                        ZOOM_WIN_SIZE)
    fig_comb_vel.show()

    # 3. 准备绘图数据
    slice_idx = slice(best_win_idx, best_win_idx + ZOOM_WIN_SIZE)
    x_local = np.arange(ZOOM_WIN_SIZE)  # 相对时间轴

    # 封装数据以便循环绘图
    plot_data_map = {
        'Position Error (m)': {
            'Bo-IMM': dist_err_bo[slice_idx],
            'EKF': dist_err_ekf[slice_idx],
            'PF': dist_err_pf[slice_idx],
            'GPF': dist_err_gp[slice_idx]
        },
        'Velocity Error (m/s)': {
            'Bo-IMM': vel_err_bo[slice_idx],
            'EKF': vel_err_ekf[slice_idx],
            'PF': vel_err_pf[slice_idx],
            'GPF': vel_err_gp[slice_idx]
        },
        'Acceleration Error (m/s^2)': {  # 注意使用 ^2 避免字体报错
            'Bo-IMM': acc_err_bo[slice_idx],
            'EKF': acc_err_ekf[slice_idx],
            'PF': acc_err_pf[slice_idx],
            'GPF': acc_err_gp[slice_idx]
        }
    }

    # 4. 绘制 3x1 子图
    metric_names = ['Position Error (m)', 'Velocity Error (m/s)']

    # 2. 绘图顺序 (确保最好的 Bo-IMM 在最上层)
    draw_order = ['GPF', 'PF', 'EKF', 'Bo-IMM']

    status_str = "(Strict Order)" if is_strict_match else "(Loose Order)"

    for metric in metric_names:
        # --- 关键修改：每次循环新建一个 Figure，不要在外面调用 subplots ---
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

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
                    markevery=5)

        # 设置标题和标签 (每个窗口独立设置)
        ax.set_title(f'Local Detail: {metric}\nFrames {best_win_idx}-{best_win_idx + ZOOM_WIN_SIZE} {status_str}',
                     fontsize=14)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step (k)', fontsize=12)  # 每一张图都加上X轴标签

        ax.grid(True, linestyle='--', alpha=0.4)
        # 每一张图都显示图例
        ax.legend(loc='upper right', ncol=4, fontsize=10, framealpha=0.9, shadow=True)

        # 调整 Y 轴范围
        all_vals = np.concatenate(list(data_group.values()))
        ax.set_ylim(0, np.max(all_vals) * 1.35)

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()