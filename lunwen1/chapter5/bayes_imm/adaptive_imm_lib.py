import numpy as np
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced


class JilkovAdaptiveIMM(IMMFilterEnhanced):
    """
    基于 Jilkov (2004) 的在线自适应 IMM。
    它不需要贝叶斯优化，而是通过统计特性自适应更新转移矩阵。
    可直接作为 IMMFilterEnhanced 的子类使用。
    """

    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None, window_len=40.0):
        super().__init__(transition_probabilities, initial_state, initial_cov, r_cov)

        # --- Jilkov 算法专用参数 ---
        # 窗口长度决定了遗忘因子，window_len=40 意味着主要关注最近 40 帧的数据
        self.window_len = window_len
        self.epsilon = 1.0 / window_len

        # 初始化“伪计数矩阵” N_ij
        # 初始时刻，我们假设当前的 trans_prob 是基于 window_len 这么多样本得来的
        self.N_counts = self.trans_prob * self.window_len

        # 用于存储交互步骤计算出的混合概率 P(M(k-1)=i | M(k)=j)
        self.mixing_probs = np.zeros((self.M, self.M))

    def interact(self):
        """
        重写 interact 方法。
        逻辑与父类完全一致，唯一的区别是：
        我们需要把中间变量 mixing_probs 存到 self.mixing_probs 中，
        供 update 步骤计算自适应更新量使用。
        """
        self.c_bar = np.dot(self.trans_prob.T, self.model_probs)
        EPS = 1e-12

        # mixing_probs[i, j] = P(M(k-1)=i | M(k)=j)
        # 这是 Jilkov 算法的核心依据
        current_mixing_probs = (self.trans_prob * self.model_probs[:, None]) / (self.c_bar + EPS)

        # 【关键】保存下来
        self.mixing_probs = current_mixing_probs.copy()

        x_mixed = np.zeros_like(self.x)
        P_mixed = np.zeros_like(self.P)

        for j in range(self.M):
            for i in range(self.M):
                x_mixed[j] += current_mixing_probs[i, j] * self.x[i]
            for i in range(self.M):
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += current_mixing_probs[i, j] * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed

    def update(self, z, dt):
        """
        重写 update 方法。
        先调用父类完成标准滤波，然后利用新信息更新转移矩阵。
        """
        # 1. 执行标准的 IMM 更新 (得到新的 self.model_probs 和 x_out)
        x_out, likelihood = super().update(z, dt)

        # 2. Jilkov 自适应更新步骤
        # 逻辑：利用后验概率 self.model_probs 和 混合概率 self.mixing_probs
        # 推断刚才发生了什么转移，并增加对应的计数。

        for j in range(self.M):
            for i in range(self.M):
                # 计算“软增量”：
                # 当前在模型 j 且是从模型 i 转移过来的概率质量
                # Increment = P(M(k)=j | Z^k) * P(M(k-1)=i | M(k)=j)
                increment = self.model_probs[j] * self.mixing_probs[i, j]

                # 递归更新计数矩阵 N (带遗忘因子)
                self.N_counts[i, j] = (1 - self.epsilon) * self.N_counts[i, j] + increment

        # 3. 归一化 N_counts 得到新的转移矩阵
        # Pi_ij = N_ij / sum_k(N_ik)
        for i in range(self.M):
            row_sum = np.sum(self.N_counts[i, :])
            if row_sum > 1e-12:
                self.trans_prob[i, :] = self.N_counts[i, :] / row_sum
            else:
                self.trans_prob[i, :] = np.ones(self.M) / self.M

        return x_out, likelihood