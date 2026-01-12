import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced

# ==========================================
# 1. 配置参数
# ==========================================
CSV_FILE_PATH = r'../../../dataSet/test_data/f16_complex_data_spiral.csv'
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

def create_trans_matrix(diag_val):
    """创建对角线占优的转移矩阵"""
    p = diag_val
    off = (1.0 - p) / 2.0
    return np.array([
        [p, off, off],
        [off, p, off],
        [off, off, p]
    ])

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
    # a, b, c, d, e, f = 0.989, 0.01, 0.11648423, 0.88251576, 0.86352624, 0.01
    trans_pa = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [e, f, 1 - e - f]
    ])

    # 4. 初始化滤波器
    print("正在初始化滤波器...")
    imm_bo = IMMFilterEnhanced(create_trans_matrix(0.975), initial_state, initial_cov, r_cov=r_cov)
    imm_06 = IMMFilterEnhanced(create_trans_matrix(0.997), initial_state, initial_cov, r_cov=r_cov)
    imm_08 = IMMFilterEnhanced(create_trans_matrix(0.9999), initial_state, initial_cov, r_cov=r_cov)
    imm_098 = IMMFilterEnhanced(create_trans_matrix(0.999999), initial_state, initial_cov, r_cov=r_cov)

    # 5. 运行滤波
    print("正在运行 Bo-IMM ...")
    est_bo, probs_bo = run_filter(imm_bo, meas_pos, DT)

    print("正在运行 0.6-IMM ...")
    est_06, probs_06 = run_filter(imm_06, meas_pos, DT)

    print("正在运行 0.8-IMM ...")
    est_08, probs_08 = run_filter(imm_08, meas_pos, DT)

    print("正在运行 0.98-IMM ...")
    est_098, probs_098 = run_filter(imm_098, meas_pos, DT)

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
    dist_err_06, vel_err_06, acc_err_06 = calc_true_metrics(est_06)
    dist_err_08, vel_err_08, acc_err_08 = calc_true_metrics(est_08)
    dist_err_098, vel_err_098, acc_err_098 = calc_true_metrics(est_098)

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
    print_stats("0.6-IMM", dist_err_06, vel_err_06, acc_err_06)
    print_stats("0.8-IMM", dist_err_08, vel_err_08, acc_err_08)
    print_stats("0.98-IMM", dist_err_098, vel_err_098, acc_err_098)
    print("-" * 100)

    # ==========================================
    # 7. 绘图 (严格还原原始颜色和样式)
    # ==========================================
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

        # 确保时间轴存在
    t_axis = np.arange(num_steps) * DT

    # -------------------------------------------------------
    # 配置绘图数据
    # -------------------------------------------------------
    plot_configs = [
        (probs_bo, "0.98-IMM"),
        (probs_06, "BO-IMM"),
        (probs_08, "bayesOnline"),
        (probs_098, "NN-IMM")
    ]

    # --- [样式修改] 参考 main_nn_inference.py 的配色 ---
    # 对应关系：
    # 索引0 (Model 1/CV): 绿色(g), alpha=0.6, lw=1.5
    # 索引1 (Model 2/CA): 蓝色(b), alpha=0.6, lw=1.5
    # 索引2 (Model 3/CT): 红色(r), alpha=1.0, lw=2.0 (重点加粗)

    styles = [
        # m=0: CV Prob (绿色，半透明)
        {'label': 'CV Prob', 'color': 'g', 'alpha': 1.0, 'lw': 2.0, 'ls': '-', 'zorder': 2},

        # m=1: CA Prob (蓝色，半透明)
        {'label': 'CA Prob', 'color': 'b', 'alpha': 1.0, 'lw': 2.0, 'ls': '-', 'zorder': 1},

        # m=2: CT Prob (红色，不透明，加粗，最上层)
        {'label': 'CT Prob', 'color': 'r', 'alpha': 1.0, 'lw': 2.0, 'ls': '-', 'zorder': 3}
    ]

    # -------------------------------------------------------
    # 循环绘制：每个模型单独一个 Figure
    # -------------------------------------------------------
    for probs, title in plot_configs:
        plt.figure(figsize=(12, 5))  # 调整为与 main_nn 类似的宽长比

        for m in range(3):
            plt.plot(t_axis, probs[m, :],
                     label=styles[m]['label'],
                     color=styles[m]['color'],
                     linewidth=styles[m]['lw'],
                     linestyle=styles[m]['ls'],
                     alpha=styles[m]['alpha'],
                     zorder=styles[m]['zorder'])

        plt.title(f'{title} Model Probability Evolution', fontsize=14, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.ylabel('Probability')
        plt.xlabel('Time (s)')

        # 样式修改：虚线网格
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

    # ==========================================
    # 8. 保存模型概率数据到 .npz
    # ==========================================
    save_path = 'model_probs.npz'
    print(f"正在保存数据到 {save_path} ...")

    np.savez(save_path,
             t=t_axis,  # 时间轴
             probs_098=probs_bo,  # Bo-IMM 的概率数据
             probs_bo=probs_06,  # 0.6-IMM 的概率数据
             probs_bayes=probs_08,  # 0.8-IMM 的概率数据
             probs_nn=probs_098  # 0.98-IMM 的概率数据
             )

    print(">>> 数据保存完成！")
    # 显示所有窗口
    plt.show()

if __name__ == "__main__":
    main()