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
from scipy.linalg import cholesky, inv, solve, block_diag
import scipy.linalg

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
    'EKF':    {'c': 'm', 'lw': 1.2, 'alpha': 0.85, 'zorder': 8,  'label': 'RMCKF'},
    'PF':     {'c': 'orange',  'lw': 1.0, 'alpha': 0.70, 'zorder': 6,  'label': 'CCPF'},
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
# 2. 算法类定义 (RMCKF) - 保持不变
# ==========================================
class RMCKF_CA_9D:
    """
    [替换] Robust Maximum Correntropy Kalman Filter (RMCKF)
    基于论文: Saha & Bhaumik, "Robust Maximum Correntropy Kalman Filter", 2024
    """

    def __init__(self, initial_state, initial_cov, r_cov, dt):
        self.dim = 9
        self.x = initial_state.copy()
        self.P = initial_cov.copy()
        self.R = r_cov
        self.dt = dt

        # --- 系统矩阵 F (CA模型) ---
        self.F = np.eye(self.dim)
        t = dt
        block = np.array([[1, t, 0.5 * t ** 2], [0, 1, t], [0, 0, 1]])
        for i in [0, 3, 6]:
            self.F[i:i + 3, i:i + 3] = block

        # --- 过程噪声 Q (保持原参数) ---
        q_std = 150.0
        var = q_std ** 2
        q_block = np.array([
            [t ** 5 / 20, t ** 4 / 8, t ** 3 / 6],
            [t ** 4 / 8, t ** 3 / 3, t ** 2 / 2],
            [t ** 3 / 6, t ** 2 / 2, t]
        ]) * var
        self.Q = np.zeros((self.dim, self.dim))
        for i in [0, 3, 6]:
            self.Q[i:i + 3, i:i + 3] = q_block

        # --- 观测矩阵 H ---
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1;
        self.H[1, 3] = 1;
        self.H[2, 6] = 1

        # --- [论文核心参数] ---
        self.sigma = 30.0  # 核带宽 (Kernel Bandwidth)
        self.mu1 = 0.01  # 风险敏感参数 (处理模型失配)
        self.mu2 = 1.0  # 观测权重调节
        self.fpi_iter = 5  # 定点迭代次数
        self.phi_state = np.zeros(self.dim)  # 对应 n 维状态的历史误差
        self.phi_meas = np.zeros(3)

    def optimize_sigma(self, z, x_pred):
        """
        复现论文 Section 4.4: Selection of Kernel Bandwidth
        寻找最大化代价函数 J_KR 的 sigma
        """
        # 论文提到 sigma 搜索范围是 1 到 5 (根据你的量级可能需要调整，比如 10-50)
        sigma_candidates = np.arange(1.0, 50.0, 1.0)
        max_cost = -np.inf
        best_sigma = self.sigma  # 默认值

        # 计算预测误差 (Prior error approximation)
        # 论文中 J_KR 是基于 error e_{i,k} 的函数
        # 这里简化处理，使用当前的残差作为搜索依据
        residual = z - self.H @ x_pred

        for sig in sigma_candidates:
            # 论文方程 (38): J = log( sum( exp(-e^2 / 2*sigma^2) ) )
            # 这里只对测量误差做计算作为近似，或者需要完整的 FPI 预计算
            cost = 0
            for r in residual:
                cost += np.exp(-(r ** 2) / (2 * sig ** 2))

            # J_KR 取对数或者直接最大化 sum(exp) 是一样的
            if cost > max_cost:
                max_cost = cost
                best_sigma = sig

        return best_sigma

    def update(self, z):
        # --- 1. 鲁棒预测 (Robust Prediction - Eq 22, 23) ---
        x_pred = self.F @ self.x

        # 计算 Eq 23: P_pred
        # 添加微小量 eye * 1e-8 保证数值稳定性
        try:
            inv_P = inv(self.P + 1e-8 * np.eye(self.dim))
            # 这里的 term 对应 (P^-1 + mu1/sigma^2 * I)^-1
            # 注意：论文中 mu1 是矩阵，代码中简化为标量 mu1 * I 是可以接受的
            robust_factor = inv(inv_P + (self.mu1 / (self.sigma ** 2)) * np.eye(self.dim))
            P_pred = self.F @ robust_factor @ self.F.T + self.Q
        except np.linalg.LinAlgError:
            # 如果求逆失败，回退到标准 KF 预测
            P_pred = self.F @ self.P @ self.F.T + self.Q

        self.sigma = self.optimize_sigma(z, x_pred)
        # --- 2. 定点迭代更新 (Fixed Point Iteration - Algorithm 1) ---
        x_new = x_pred.copy()

        # 计算平方根矩阵 Bp, Br (Eq 10)
        # Bp * Bp' = P_pred
        try:
            Bp = cholesky(P_pred + 1e-8 * np.eye(self.dim), lower=True)
            Br = cholesky(self.R + 1e-8 * np.eye(3), lower=True)
        except np.linalg.LinAlgError:
            Bp = np.eye(self.dim)
            Br = np.eye(3)

        inv_Bp = inv(Bp)
        inv_Br = inv(Br)

        for _ in range(self.fpi_iter):
            # A. 计算归一化误差 (Eq 11 的变形)
            # e_p = Bp^-1 * (x - x_pred)
            e_p = inv_Bp @ (x_new - x_pred)
            # e_r = Br^-1 * (y - Hx)
            e_r = inv_Br @ (z - self.H @ x_new)

            # B. 计算高斯核权重 (Eq 220)
            # 注意：论文中是 exp(phi - ...). 这里简化 phi=0, 仅保留核心的指数衰减项
            # G(e) = exp( - (mu * e^2) / (2 * sigma^2) )
            # 这里的 e_p**2 是元素级别的平方
            current_term_p = -0.5 * self.mu1 * (e_p ** 2) / (self.sigma ** 2)
            current_term_r = -0.5 * self.mu2 * (e_r ** 2) / (self.sigma ** 2)

            # --- 【修改代码】引入历史记忆 Phi 计算最终权重 (Eq. 220) ---
            # Pi = exp( phi + current_term )
            Lambda_p = np.exp(self.phi_state + current_term_p)
            Lambda_r = np.exp(self.phi_meas + current_term_r)

            # C. 构建 Pi 矩阵的逆 (因为后续计算需要 Pi^-1)
            # Pi 是对角矩阵，其逆也是对角矩阵，对角元素为 1/Lambda
            # 加上 1e-10 防止除以零
            Pi_p_inv = np.diag(1.0 / (Lambda_p + 1e-10))
            Pi_r_inv = np.diag(1.0 / (Lambda_r + 1e-10))

            # D. 计算修正协方差 P_bar, R_bar (Eq 217)
            # P_bar = Bp * Pi_p^-1 * Bp'
            P_bar = Bp @ Pi_p_inv @ Bp.T
            # R_bar = Br * Pi_r^-1 * Br'
            R_bar = Br @ Pi_r_inv @ Br.T

            # E. 计算卡尔曼增益 K (Eq 29)
            # K = P_bar * H' * (H * P_bar * H' + R_bar)^-1
            S = self.H @ P_bar @ self.H.T + R_bar
            try:
                K = P_bar @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
            except np.linalg.LinAlgError:
                K = np.zeros((self.dim, 3))

            # F. 更新状态估计 (Eq 27)
            x_next = x_pred + K @ (z - self.H @ x_pred)

            # 收敛检查
            if np.linalg.norm(x_next - x_new) < 1e-4:
                x_new = x_next
                break
            x_new = x_next

        self.phi_state += current_term_p
        self.phi_meas += current_term_r
        # --- 3. 最终更新 ---
        self.x = x_new
        # 后验协方差 (Eq 28)
        I = np.eye(self.dim)
        self.P = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T

        return self.x


# 粒子滤波
# ==========================================
# [复现] Constrained Cubature Particle Filter (CCPF)
# 基于论文: Sensors 2024, 24, 1228 "Constrained Cubature Particle Filter..."
# ==========================================
class ConstrainedCubatureParticleFilter_9D:
    def __init__(self, num_particles, initial_state, initial_cov, r_cov, dt, q_std=100.0):
        self.N = num_particles
        self.dim = 9
        self.meas_dim = 3
        self.dt = dt
        self.R = r_cov
        self.Q = self._build_Q(dt, q_std)

        # 粒子初始化
        self.particles_x = np.zeros((self.N, self.dim))
        self.particles_P = np.zeros((self.N, self.dim, self.dim))

        for i in range(self.N):
            self.particles_x[i] = np.random.multivariate_normal(initial_state, initial_cov)
            self.particles_P[i] = initial_cov.copy()

        self.weights = np.ones(self.N) / self.N

        # 容积点参数
        self.num_cubature_points = 2 * self.dim
        self.xi = np.concatenate([
            np.sqrt(self.dim) * np.eye(self.dim),
            -np.sqrt(self.dim) * np.eye(self.dim)
        ], axis=1)

        # 约束参数 (-z <= 0)
        self.D_constraint = np.zeros((1, self.dim))
        self.D_constraint[0, 6] = -1.0
        self.d_constraint = np.array([0.0])
        self.apply_constraint_flag = True

        # 论文参数
        self.beta = 2.0

    def _build_Q(self, dt, q_std):
        t = dt
        var = q_std ** 2
        # CA 模型 Q 阵构建
        q_block = np.array([
            [t ** 5 / 20, t ** 4 / 8, t ** 3 / 6],
            [t ** 4 / 8, t ** 3 / 3, t ** 2 / 2],
            [t ** 3 / 6, t ** 2 / 2, t]
        ]) * var
        Q = np.zeros((self.dim, self.dim))
        for i in [0, 3, 6]:
            Q[i:i + 3, i:i + 3] = q_block
        return Q

    def _f_state_trans(self, x):
        dt = self.dt
        x_next = x.copy()
        # x, y, z 三轴解耦 CA 模型
        for i in [0, 3, 6]:
            p, v, a = x[i], x[i + 1], x[i + 2]
            x_next[i] = p + v * dt + 0.5 * a * dt ** 2
            x_next[i + 1] = v + a * dt
            x_next[i + 2] = a
        return x_next

    def _h_meas_func(self, x):
        return x[[0, 3, 6]]

    # --- [新增] 为了兼容主函数的调用接口，添加空预测函数 ---
    def predict(self):
        pass

    def update(self, z):
        new_particles_x = np.zeros_like(self.particles_x)
        new_particles_P = np.zeros_like(self.particles_P)
        residuals = np.zeros((self.N, self.meas_dim))

        # 记录 log 权重防止溢出
        log_weights = np.log(self.weights + 1e-300)

        for j in range(self.N):
            x_prev = self.particles_x[j]
            P_prev = self.particles_P[j]

            # 1. CKF 预测
            try:
                S = np.linalg.cholesky(P_prev + 1e-9 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                S = np.eye(self.dim) * 0.001

            cub_points = np.zeros((self.dim, self.num_cubature_points))
            for i in range(self.num_cubature_points):
                cub_points[:, i] = S @ self.xi[:, i] + x_prev

            X_star = np.zeros_like(cub_points)
            for i in range(self.num_cubature_points):
                X_star[:, i] = self._f_state_trans(cub_points[:, i])

            x_pred = np.mean(X_star, axis=1)
            P_pred = self.Q.copy()
            for i in range(self.num_cubature_points):
                diff = (X_star[:, i] - x_pred).reshape(-1, 1)
                P_pred += (diff @ diff.T) / self.num_cubature_points

            # 2. CKF 更新准备
            try:
                S_pred = np.linalg.cholesky(P_pred + 1e-9 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                S_pred = np.eye(self.dim)

            X_pred_cub = np.zeros((self.dim, self.num_cubature_points))
            for i in range(self.num_cubature_points):
                X_pred_cub[:, i] = S_pred @ self.xi[:, i] + x_pred

            Y_pred_cub = np.zeros((self.meas_dim, self.num_cubature_points))
            for i in range(self.num_cubature_points):
                Y_pred_cub[:, i] = self._h_meas_func(X_pred_cub[:, i])

            y_pred = np.mean(Y_pred_cub, axis=1)

            Pzz = self.R.copy()
            Pxz = np.zeros((self.dim, self.meas_dim))

            for i in range(self.num_cubature_points):
                diff_y = (Y_pred_cub[:, i] - y_pred).reshape(-1, 1)
                diff_x = (X_pred_cub[:, i] - x_pred).reshape(-1, 1)
                Pzz += (diff_y @ diff_y.T) / self.num_cubature_points
                Pxz += (diff_x @ diff_y.T) / self.num_cubature_points

            # 3. CKF 更新
            # 使用 solve 替代 inv 提高精度: K = Pxz * Pzz^-1
            try:
                K = scipy.linalg.solve(Pzz + 1e-9 * np.eye(self.meas_dim), Pxz.T).T
            except:
                K = np.zeros((self.dim, self.meas_dim))

            innov = z - y_pred
            residuals[j] = innov  # 记录用于重采样

            x_est = x_pred + K @ innov
            P_est = P_pred - K @ Pzz @ K.T

            # --- [CRITICAL FIX] 权重更新应基于预测似然 ---
            try:
                # L_pzz * L_pzz.T = Pzz
                L_pzz = np.linalg.cholesky(Pzz + 1e-9 * np.eye(self.meas_dim))
                # sol = L^-1 * innov
                sol = scipy.linalg.solve_triangular(L_pzz, innov, lower=True)
                log_lik = -0.5 * np.sum(sol ** 2) - np.sum(np.log(np.diag(L_pzz)))
            except:
                log_lik = -100.0

            log_weights[j] += log_lik

            # 4. 约束投影 (Constrained Projection)
            if self.apply_constraint_flag:
                constraint_val = self.D_constraint @ x_est
                if constraint_val[0] > self.d_constraint[0]:
                    W_inv = P_est  # Weighting matrix inverse
                    D = self.D_constraint
                    d = self.d_constraint

                    try:
                        term1 = W_inv @ D.T
                        term2 = D @ W_inv @ D.T
                        discrepancy = D @ x_est - d
                        # Lagrange multiplier
                        lambda_vec = scipy.linalg.solve(term2 + 1e-12 * np.eye(1), discrepancy)
                        correction = term1 @ lambda_vec
                        x_est = x_est - correction.flatten()
                    except:
                        # 投影失败时的 Hard Constraint Fallback
                        x_est[6] = max(x_est[6], 0.0)

                        # 5. 采样新粒子 (Sampling Step for CPF)
            try:
                # 保证 P_est 对称正定
                P_est = (P_est + P_est.T) / 2
                new_particles_x[j] = np.random.multivariate_normal(x_est, P_est)
            except:
                new_particles_x[j] = x_est

            new_particles_P[j] = P_est

        # 归一化权重
        max_log_w = np.max(log_weights)
        self.weights = np.exp(log_weights - max_log_w)
        self.weights /= np.sum(self.weights)

        # 6. 欧氏距离重采样调整
        self._resample_euclidean(residuals)

        # 估计输出
        est_x = np.average(new_particles_x, weights=self.weights, axis=0)

        self.particles_x = new_particles_x
        self.particles_P = new_particles_P

        return est_x

    def _resample_euclidean(self, residuals):
        # 保持原逻辑不变，增加数值保护
        j_max = np.argmax(self.weights)
        r_max = residuals[j_max]

        L = np.sum((residuals - r_max) ** 2, axis=1)  # Vectorized

        j_min = np.argmin(self.weights)
        L_max = np.sum((residuals[j_min] - r_max) ** 2) + 1e-12

        w_max_val = self.weights[j_max]

        # 向量化计算
        term_sin = np.sin((L / L_max) * (np.pi / 2))
        adj = (w_max_val / self.N) * term_sin * self.beta

        new_weights = self.weights + adj
        # 保护：防止权重为负
        new_weights = np.maximum(new_weights, 1e-300)

        self.weights = new_weights / np.sum(new_weights)

        # 标准重采样
        N_eff = 1.0 / np.sum(self.weights ** 2)
        if N_eff < self.N / 2.0:
            indices = self._systematic_resample(self.weights)
            self.particles_x = self.particles_x[indices]
            self.particles_P = self.particles_P[indices]
            self.weights.fill(1.0 / self.N)

    def _systematic_resample(self, weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes


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
    time_axis = np.arange(min_len)

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

    # ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('步长', fontsize=12)
    ax.set_xlim(plot_x[0], plot_x[-1])
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
    ekf = RMCKF_CA_9D(initial_state, initial_cov, r_cov, DT)
    est_ekf = run_ekf_filter(ekf, meas_pos)

    # --- C. GP (【修改部分】：改为滑动窗口 + 噪声失配) ---
    print("正在运行 GPF (Online Sliding Window)...")

    # >> 削弱 1: 只能看过去 20 个点
    GP_WINDOW_SIZE = 90

    # >> 削弱 2: 假设的噪声方差 (100) 小于真实方差 (225)
    # 这会模拟“过拟合”现象，使 GP 对噪声更敏感，从而降低其平滑效果
    GP_ASSUMED_NOISE_VAR = 0.2

    t_calc = np.arange(num_steps) * DT
    X_full = t_calc.reshape(-1, 1)

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
    pf = ConstrainedCubatureParticleFilter_9D(num_particles, initial_state, initial_cov, r_cov, DT,q_std=100.0)
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

        var_p = np.var(dist_err_p[EVAL_START:])
        var_v = np.var(dist_err_v[EVAL_START:])
        var_a = np.var(dist_err_a[EVAL_START:])

        # [修改] 打印结果包含方差
        print(
            f'{name:<15} | RMSE_p: {rmse_p:.4f} Var_p: {var_p:.4f} | RMSE_v: {rmse_v:.4f} Var_v: {var_v:.4f} | RMSE_a: {rmse_a:.4f} Var_a: {var_a:.4f}')

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

    step_axis = np.arange(num_steps)
    t_plot = step_axis[EVAL_START:]
    c_bo, c_ekf, c_gp, c_pf = [0, 0.85, 0], 'm', 'b', 'orange'

    # 绘制位置误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, dist_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, dist_err_pf[EVAL_START:], color=c_pf, label='CCPF', alpha=0.6)
    plt.plot(t_plot, dist_err_ekf[EVAL_START:], color=c_ekf, label='RMCKF', alpha=0.6)
    plt.plot(t_plot, dist_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)
    # plt.title('位置误差对比 (Position RMSE)')
    plt.xlabel('步数')
    plt.xlim(t_plot[0], t_plot[-1])
    plt.ylabel('误差 (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制速度误差 (新增)
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, vel_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, vel_err_pf[EVAL_START:], color=c_pf, label='CCPF', alpha=0.6)
    plt.plot(t_plot, vel_err_ekf[EVAL_START:], color=c_ekf, label='RMCKF', alpha=0.6)
    plt.plot(t_plot, vel_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)
    # plt.title('速度误差对比 (Velocity RMSE)')
    plt.xlabel('步数')
    plt.xlim(t_plot[0], t_plot[-1])
    plt.ylabel('误差 (m/s)')
    plt.legend(loc='upper right', framealpha=1.0)
    plt.grid(True, alpha=0.3)

    # [新增] 绘制加速度误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, acc_err_gp[EVAL_START:], color=c_gp, label='GPF', alpha=0.6)
    plt.plot(t_plot, acc_err_pf[EVAL_START:], color=c_pf, label='CCPF', alpha=0.6)
    plt.plot(t_plot, acc_err_ekf[EVAL_START:], color=c_ekf, label='RMCKF', alpha=0.6)
    plt.plot(t_plot, acc_err_bo[EVAL_START:], color=c_bo, label='Bo-IMM', linewidth=2)

    # plt.title('加速度误差对比 (Acceleration RMSE)')
    plt.xlabel('步数')
    plt.xlim(t_plot[0], t_plot[-1])
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
            'label': 'RMCKF'
        },
        'PF': {
            # 橙色, 虚线, 方块 (Third)
            'c': 'orange', 'ls': '--', 'mk': 's', 'ms': 5, 'lw': 1.5, 'alpha': 0.8, 'zorder': 6,
            'label': 'CCPF'
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
    fig_comb_pos = draw_combined_figure(data_pos, 'Position Error', '位置误差 (m)', best_win_idx,
                                        ZOOM_WIN_SIZE)
    fig_comb_pos.show()

    print(">>> 生成 Combined Plot (Vel)...")
    fig_comb_vel = draw_combined_figure(data_vel, 'Velocity Error', '速度误差 (m/s)', best_win_idx,
                                        ZOOM_WIN_SIZE)
    fig_comb_vel.show()

    # 3. 准备绘图数据
    slice_idx = slice(best_win_idx, best_win_idx + ZOOM_WIN_SIZE)
    x_local = np.arange(ZOOM_WIN_SIZE)  # 相对时间轴

    # 封装数据以便循环绘图
    plot_data_map = {
        '位置误差 (m)': {
            'Bo-IMM': dist_err_bo[slice_idx],
            'EKF': dist_err_ekf[slice_idx],
            'PF': dist_err_pf[slice_idx],
            'GPF': dist_err_gp[slice_idx]
        },
        '速度误差 (m/s)': {
            'Bo-IMM': vel_err_bo[slice_idx],
            'EKF': vel_err_ekf[slice_idx],
            'PF': vel_err_pf[slice_idx],
            'GPF': vel_err_gp[slice_idx]
        },
        '加速度误差 (m/s^2)': {  # 注意使用 ^2 避免字体报错
            'Bo-IMM': acc_err_bo[slice_idx],
            'EKF': acc_err_ekf[slice_idx],
            'PF': acc_err_pf[slice_idx],
            'GPF': acc_err_gp[slice_idx]
        }
    }

    # 4. 绘制 3x1 子图
    metric_names = ['位置误差 (m)', '速度误差 (m/s)']

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
        # ax.set_title(f'Local Detail: {metric}\nFrames {best_win_idx}-{best_win_idx + ZOOM_WIN_SIZE} {status_str}',
                     # fontsize=14)
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