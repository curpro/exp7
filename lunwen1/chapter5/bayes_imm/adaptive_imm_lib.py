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


class PaperCompressionRatioIMM(IMMFilterEnhanced):
    """
    论文《Adaptive IMM Algorithm Based on Variational Inference ...》(AIMM-VI) 中
    **模型概率转移矩阵自适应更新**部分的复现实现（对应论文第 3.5 节，公式 (35)-(38)）。

    注意：
    - 你要求 IMM 核心结构(三模型 CV/CA/CT、滤波更新)不能改：这里严格继承 IMMFilterEnhanced，
      只在每次 update() 结束后更新 self.trans_prob (Markov 转移矩阵)。
    - 论文里使用“增广状态”(含姿态角) + 变分推断(VI) 估计 kinematic state。
      在你的飞机点目标场景里，我们直接把你现有的 9 维状态
      [x,vx,ax, y,vy,ay, z,vz,az] 当作“增广状态”，并用各模型的 KF 更新结果作为 x_j^k。

    公式对应关系（与你的代码变量）：
    - x_j^k      -> self.x[j]          (第 j 个模型的后验状态)
    - x^k        -> x_out              (IMM 融合输出)
    - x_o,j^{k+1}-> 由当前 (x_i^k, mu_i^k, G_k) 做一次 interact/mixing 得到
    - A_j^k      = x_o,j^{k+1} - x_j^k
    - B_j^k      = x^k - x_j^k
    - lambda_j^k = ||A_j^k|| / ||B_j^k||
    - G_new(i,j) = (lambda_i/lambda_j)^l * G_old(i,j)

    参数：
    - l: 论文中的调整因子 l∈[0,1]（建议 0.2~0.8；过大可能导致矩阵震荡）
    - min_prob: 防止转移矩阵出现 0（数值稳定）
    - lambda_clip: 裁剪 lambda，避免极端比值造成数值爆炸
    """

    def __init__(self,
                 transition_probabilities,
                 initial_state,
                 initial_cov,
                 r_cov=None,
                 l: float = 0.5,
                 eps: float = 1e-12,
                 min_prob: float = 1e-6,
                 lambda_clip=(1e-3, 1e3)):
        super().__init__(transition_probabilities, initial_state, initial_cov, r_cov)
        self.l = float(l)
        self.eps = float(eps)
        self.min_prob = float(min_prob)
        self.lambda_clip = lambda_clip

    def _compute_xo_next(self) -> np.ndarray:
        """计算 x_o,j^{k+1}：使用当前时刻(k)的后验 (x_i^k, mu_i^k) 和当前 G_k 进行一次交互混合。"""
        c_bar = np.dot(self.trans_prob.T, self.model_probs)  # shape (M,)
        mixing_probs = (self.trans_prob * self.model_probs[:, None]) / (c_bar + self.eps)  # i->j conditional
        x_mixed = np.zeros_like(self.x)
        for j in range(self.M):
            for i in range(self.M):
                x_mixed[j] += mixing_probs[i, j] * self.x[i]
        return x_mixed

    def _update_transition_matrix(self, lambdas: np.ndarray):
        """按论文 Eq.(38) 更新转移矩阵，并做行归一化保证每行和为 1。"""
        lam = np.asarray(lambdas, dtype=float)

        # 裁剪，避免极端 lambda 比值导致数值爆炸
        if self.lambda_clip is not None:
            lam_min, lam_max = self.lambda_clip
            lam = np.clip(lam, lam_min, lam_max)

        G_old = self.trans_prob
        G_new = np.zeros_like(G_old)

        for i in range(self.M):
            for j in range(self.M):
                ratio = lam[i] / (lam[j] + self.eps)
                G_new[i, j] = (ratio ** self.l) * G_old[i, j]

        # 防止出现 0
        if self.min_prob is not None and self.min_prob > 0:
            G_new = np.maximum(G_new, self.min_prob)

        # 行归一化（概率转移矩阵每行和为 1）
        for i in range(self.M):
            row_sum = np.sum(G_new[i, :])
            if row_sum <= self.eps or not np.isfinite(row_sum):
                # fallback：退回旧矩阵该行
                row = np.maximum(G_old[i, :], self.min_prob)
                G_new[i, :] = row / np.sum(row)
            else:
                G_new[i, :] /= row_sum

        # 保持原数组引用不变（更稳妥）
        self.trans_prob[:] = G_new

    def update(self, z, dt):
        # 1) 先执行标准 IMM 更新（父类：不改你原有 IMM 核心结构）
        x_out, likelihood = super().update(z, dt)

        # 2) 计算 x_o,j^{k+1}
        x_o_next = self._compute_xo_next()

        # 3) 计算 lambda_j^k（公式 (35)-(36)）
        lambdas = np.zeros(self.M, dtype=float)
        for j in range(self.M):
            A = x_o_next[j] - self.x[j]
            B = x_out - self.x[j]
            lambdas[j] = np.linalg.norm(A) / (np.linalg.norm(B) + self.eps)

        # 4) 用 lambda 更新转移矩阵（公式 (38)）
        self._update_transition_matrix(lambdas)

        return x_out, likelihood

