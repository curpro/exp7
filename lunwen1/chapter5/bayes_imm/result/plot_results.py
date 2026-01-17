import numpy as np
import matplotlib.pyplot as plt

# =================配置=================
# 这里只需要和绘图相关的配置
DT = 1 / 30


# ======================================

def analyze_and_plot():
    file_path = r'D:\AFS\lunwen\lunwen1\chapter5\bayes_imm\result\Opt__10\r_300_20\imm_results_90.npz'

    print(f"=== 正在读取本地数据: {file_path} ===")
    try:
        data = np.load(file_path)
        t = data['t']
        err_fixed_pos = data['err_fixed_pos']
        err_online_pos = data['err_online_pos']
        err_fixed_vel = data['err_fixed_vel']
        err_online_vel = data['err_online_vel']
        param_history = data['param_history']
        print("数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请先运行 main_online.py 生成数据。")
        return

    # ================= 数据统计与打印 (逻辑与原文件一致) =================
    eval_start_idx = 90  # 评估起始帧

    # 1. 计算平均 RMSE
    rmse_pos_fix_val = np.sqrt(np.mean(err_fixed_pos[eval_start_idx:] ** 2))
    rmse_pos_onl_val = np.sqrt(np.mean(err_online_pos[eval_start_idx:] ** 2))
    rmse_vel_fix_val = np.sqrt(np.mean(err_fixed_vel[eval_start_idx:] ** 2))
    rmse_vel_onl_val = np.sqrt(np.mean(err_online_vel[eval_start_idx:] ** 2))

    # 2. 计算误差方差
    var_pos_fix_val = np.var(err_fixed_pos[eval_start_idx:])
    var_pos_onl_val = np.var(err_online_pos[eval_start_idx:])
    var_vel_fix_val = np.var(err_fixed_vel[eval_start_idx:])
    var_vel_onl_val = np.var(err_online_vel[eval_start_idx:])

    # 3. 计算最大误差
    max_pos_fix_val = np.max(err_fixed_pos[eval_start_idx:])
    max_pos_onl_val = np.max(err_online_pos[eval_start_idx:])
    max_vel_fix_val = np.max(err_fixed_vel[eval_start_idx:])
    max_vel_onl_val = np.max(err_online_vel[eval_start_idx:])

    # 4. 计算提升率
    pos_rmse_improv = (rmse_pos_fix_val - rmse_pos_onl_val) / rmse_pos_fix_val * 100
    vel_rmse_improv = (rmse_vel_fix_val - rmse_vel_onl_val) / rmse_vel_fix_val * 100
    pos_var_improv = (var_pos_fix_val - var_pos_onl_val) / var_pos_fix_val * 100
    vel_var_improv = (var_vel_fix_val - var_vel_onl_val) / var_vel_fix_val * 100
    pos_max_improv = (max_pos_fix_val - max_pos_onl_val) / max_pos_fix_val * 100
    vel_max_improv = (max_vel_fix_val - max_vel_onl_val) / max_vel_fix_val * 100

    # 5. 打印表格
    print("\n" + "=" * 70)
    print("               仿真结果性能对比 (读取自本地文件)")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Fixed IMM':<12} | {'BO-IMM (Ours)':<15} | {'Improv(%)':<11}")
    print("-" * 70)
    print(f"{'Pos RMSE (m)':<18} | {rmse_pos_fix_val:<12.4f} | {rmse_pos_onl_val:<15.4f} | {pos_rmse_improv:>9.2f}%")
    print(f"{'Vel RMSE (m/s)':<18} | {rmse_vel_fix_val:<12.4f} | {rmse_vel_onl_val:<15.4f} | {vel_rmse_improv:>9.2f}%")
    print("-" * 70)
    print(f"{'Pos Var (m^2)':<18} | {var_pos_fix_val:<12.4f} | {var_pos_onl_val:<15.4f} | {pos_var_improv:>9.2f}%")
    print(f"{'Vel Var (m/s)^2':<18} | {var_vel_fix_val:<12.4f} | {var_vel_onl_val:<15.4f} | {vel_var_improv:>9.2f}%")
    print("-" * 70)
    print(f"{'Pos Max (m)':<18} | {max_pos_fix_val:<12.4f} | {max_pos_onl_val:<15.4f} | {pos_max_improv:>9.2f}%")
    print(f"{'Vel Max (m/s)':<18} | {max_vel_fix_val:<12.4f} | {max_vel_onl_val:<15.4f} | {vel_max_improv:>9.2f}%")
    print("=" * 70 + "\n")

    # ================= 绘图 1: RMSE 对比 =================
    start_plot = 90

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t[start_plot:], err_fixed_pos[start_plot:], 'b--', label='Fixed IMM', alpha=0.6)
    plt.plot(t[start_plot:], err_online_pos[start_plot:], 'r-', label='Adaptive IMM (Ours)', linewidth=1.5)
    plt.ylabel('Position RMSE (m)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Position RMSE: Fixed={rmse_pos_fix_val:.2f}m, Online={rmse_pos_onl_val:.2f}m')

    plt.subplot(2, 1, 2)
    plt.plot(t[start_plot:], err_fixed_vel[start_plot:], 'b--', label='Fixed IMM', alpha=0.6)
    plt.plot(t[start_plot:], err_online_vel[start_plot:], 'r-', label='Adaptive IMM (Ours)', linewidth=1.5)
    plt.ylabel('Velocity RMSE (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Velocity RMSE: Fixed={rmse_vel_fix_val:.2f}m/s, Online={rmse_vel_onl_val:.2f}m/s')
    plt.tight_layout()
    plt.show()

    # ================= 绘图 2: 参数变化 =================
    ph = param_history
    param_labels = [
        r'$P_{11}$ (CV$\to$CV)', r'$P_{12}$ (CV$\to$CA)',
        r'$P_{21}$ (CA$\to$CV)', r'$P_{22}$ (CA$\to$CA)',
        r'$P_{31}$ (CT$\to$CV)', r'$P_{32}$ (CT$\to$CA)'
    ]

    plt.figure(figsize=(12, 10))
    plt.suptitle('Adaptive Transition Probability Parameters (BO-IMM)')

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, ph[:, i], color='purple', linewidth=1.5, label='BO Output')
        plt.title(param_labels[i])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(-0.05, 1.05)
        if i % 2 == 0:
            plt.ylabel('Probability')
        if i >= 4:
            plt.xlabel('Time (s)')

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.show()


if __name__ == "__main__":
    analyze_and_plot()