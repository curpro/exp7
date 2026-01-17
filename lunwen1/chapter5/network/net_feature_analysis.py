import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import random

# ================= 配置 =================
MODEL_FILE = 'net/z_feature_MLP/imm_param_net.pth'
SCALER_FILE = 'net/z_feature_MLP/scaler_params.json'
DATA_PATTERN = os.path.join('npz', 'training_data_part*.npz')
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    'Pos X', 'Pos Y', 'Pos Z',
    'Vel X', 'Vel Y', 'Vel Z',
    'Acc X', 'Acc Y', 'Acc Z'
]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> 随机种子已固定为: {seed}")



# === 模型定义 (保持不变) ===
class ParamPredictorMLP(nn.Module):
    def __init__(self, seq_len=90, input_dim=9):
        super(ParamPredictorMLP, self).__init__()
        # 输入维度 = 时间步长 * 特征数 (例如 90 * 9 = 810)
        self.input_flat_dim = seq_len * input_dim

        # === 关键修正 1: 必须与训练代码完全一致 (LeakyReLU + Dropout参数) ===
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),  # [训练代码用的是 LeakyReLU]
            nn.Dropout(0.4),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),  # [训练代码用的是 LeakyReLU]
            nn.Dropout(0.3),  # [训练代码用的是 0.3]

            nn.Linear(32, 9)
        )

    def forward(self, x):
        b, s, f = x.shape
        x = x.reshape(b, -1)
        logits = self.net(x)
        logits = logits.view(-1, 3, 3)
        temperature = 2.0
        return torch.log_softmax(logits / temperature, dim=2)

    # === 关键修正 2: 必须保留这个分析用的辅助函数 ===
    def get_embedding(self, x):
        """
        提取倒数第二层的特征向量 (32维)，用于 t-SNE 可视化
        """
        b, s, f = x.shape
        x = x.reshape(b, -1)

        # 手动模拟前向传播，直到取出中间层特征
        out = x
        for i, layer in enumerate(self.net):
            out = layer(out)
            # 这里的索引对应上面的 Sequential 定义:
            # 0:Linear(810->128) -> 1:BN -> 2:LeakyReLU -> 3:Dropout
            # 4:Linear(128->32)  -> 5:BN -> 6:LeakyReLU -> 7:Dropout
            # 8:Linear(32->9)

            # 我们想要拿到 32 维的特征，通常取第 6 层 (LeakyReLU) 之后的输出
            if i == 6:
                return out
        return out


# === 数据加载 (保持不变) ===
def load_data():
    files = glob.glob(DATA_PATTERN)
    if not files:
        # 为了演示，如果没有文件生成一些假数据
        print("警告: 未找到数据文件，生成随机数据用于测试代码功能...")
        X = np.random.randn(1000, 90, 9).astype(np.float32)
        # 生成假标签: 随机概率
        Y = np.random.rand(1000, 6).astype(np.float32)
        return torch.FloatTensor(X), torch.FloatTensor(Y)

    all_X, all_Y = [], []
    print(f"Loading data from {len(files)} files...")
    for f in files:
        data = np.load(f)
        all_X.append(data['X'])
        all_Y.append(data['Y'])

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, 'r') as f:
            scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        std = np.array(scaler['std'], dtype=np.float32)
        X = (X - mean) / std

    return torch.FloatTensor(X), torch.FloatTensor(Y)


# === [核心新增 1] 梯度显著性分析 (Saliency Map) ===
def analyze_saliency_map(model, X, Y):
    print("\n>>> 1. 正在计算时空显著性图 (Saliency Map)...")
    model.eval()

    # 随机采样一部分数据
    idx = np.random.choice(len(X), min(200, len(X)), replace=False)
    inputs = X[idx].to(DEVICE)
    inputs.requires_grad = True  # 关键：开启输入梯度

    # 前向传播
    log_probs = model(inputs)  # (B, 3, 3)

    # 我们不仅想知道loss对输入的梯度，更想知道“预测结果”对输入的梯度
    # 这里我们最大化预测概率最大的那个类别的输出
    # 简单的做法：对 output 求 sum 也可以得到梯度幅度
    batch_size = inputs.shape[0]
    flat_log_probs = log_probs.view(batch_size, -1)

    # 2. 找到每一行最大的那个值 (Best Score)
    # values 是最大的分数，indices 是它是第几个类别
    max_scores, indices = torch.max(flat_log_probs, dim=1)

    # 3. 只对这些“最大值”求和并反向传播
    # 含义：我想知道是哪些特征让这些“最大值”变得这么大的？
    target_score = max_scores.sum()

    model.zero_grad()
    target_score.backward()

    # 获取梯度：(B, 90, 9)
    grads = inputs.grad.data.cpu().numpy()

    # 取绝对值并平均 (Magnitude of gradients)
    # (90, 9) -> 时间步 x 特征
    saliency = np.mean(np.abs(grads), axis=0)

    # saliency = saliency[:75, :]

    # 绘图
    plt.figure(figsize=(12, 6))  # 保持原本的宽图（适合时间序列）

    # 1. 绘制热力图
    # 保持使用 saliency.T (特征在行/Y轴，时间在列/X轴)
    ax = sns.heatmap(saliency.T, cmap='viridis',
                     xticklabels=10,
                     yticklabels=FEATURE_NAMES,
                     cbar_kws={'label': 'Gradient Magnitude'})

    # === 关键修改 1: 强制 Y 轴文字水平显示 (横过来) ===
    # rotation=0 表示文字水平放置
    plt.yticks(rotation=0)

    # === 关键修改 2: 将 Colorbar 标签移到左侧 ===
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_label_position('left')

    # 坐标轴标签
    plt.xlabel('Time Step (k)')
    plt.ylabel('Input Features')
    plt.title('Spatiotemporal Saliency Map', fontsize=14)

    plt.tight_layout()
    plt.show()

    return saliency


# === [核心新增 2] t-SNE 隐空间可视化 ===
def analyze_tsne_distribution(model, X, Y):
    print("\n>>> 2. 正在计算 t-SNE 分布 (挖掘模式: 稀有优先)...")

    # === [步骤 1] 智能标签提取与判定 ===
    Y_np = Y.numpy()

    # 1.1 自动解析概率列 (兼容 6列/9列/3列 格式)
    if Y_np.shape[1] == 6:
        # [p11, p12, p21, p22, p31, p32]
        p_cv = Y_np[:, 0]  # p11
        p_ca = Y_np[:, 3]  # p22
        p_ct = 1.0 - Y_np[:, 4] - Y_np[:, 5]  # p33
    elif Y_np.shape[1] == 9:
        # 3x3 Flatten
        p_cv = Y_np[:, 0]
        p_ca = Y_np[:, 4]
        p_ct = Y_np[:, 8]
    elif Y_np.shape[1] == 3:
        # [cv, ca, ct]
        p_cv = Y_np[:, 0]
        p_ca = Y_np[:, 1]
        p_ct = Y_np[:, 2]
    else:
        print(f"[Error] 未知的标签形状: {Y_np.shape}")
        return

    # 1.2 [核心修改] 优先级判定逻辑
    # 默认全是 0 (CV/直行)
    labels = np.zeros(len(Y_np), dtype=int)

    # 阈值判定：只要概率超过 0.45，就认定为机动
    # 注意顺序：先判 CA，再判 CT (让 CT 覆盖 CA，或者反过来，视谁更稀有而定)
    # 这里我们认为 CT(转弯) 最稀有，给最高优先级

    # 找出所有“疑似”样本
    mask_ca = p_ca > 0.6
    mask_ct = p_ct > 0.6

    labels[mask_ca] = 1  # 标记为 CA
    labels[mask_ct] = 2  # 标记为 CT (如果同时满足，CT会覆盖CA)

    # 1.3 统计数量
    idx_cv = np.where(labels == 0)[0]
    idx_ca = np.where(labels == 1)[0]
    idx_ct = np.where(labels == 2)[0]

    print("\n" + "=" * 50)
    print("   [数据集深度诊断]")
    print(f"   样本总数: {len(Y_np)}")
    print(f"   CV (直行/背景): {len(idx_cv)} 样本")
    print(f"   CA (加速/机动): {len(idx_ca)} 样本 (阈值>0.45)")
    print(f"   CT (转弯/机动): {len(idx_ct)} 样本 (阈值>0.45)")
    print("=" * 50)

    # === [步骤 2] 平衡采样 (防止 CV 点太多把图撑爆) ===
    samples_per_class = 300  # 每类最多取 300 个点

    final_idx = []

    # 定义采样函数
    def safe_sample(indices, n):
        if len(indices) == 0: return []
        if len(indices) <= n: return indices  # 不够就全拿
        return np.random.choice(indices, n, replace=False)

    # 分别采样
    final_idx.extend(safe_sample(idx_cv, samples_per_class))
    final_idx.extend(safe_sample(idx_ca, samples_per_class))
    final_idx.extend(safe_sample(idx_ct, samples_per_class))

    final_idx = np.array(final_idx, dtype=int)
    np.random.shuffle(final_idx)  # 打乱顺序

    print(f"   [绘图准备] 最终采样用于 t-SNE 的点数: {len(final_idx)}")

    if len(final_idx) < 10:
        print("   [警告] 样本太少，无法绘制 t-SNE。")
        return

    # === [步骤 3] 提取特征 (Embedding) ===
    X_sub = X[final_idx].to(DEVICE if 'DEVICE' in globals() else 'cpu')
    model.eval()

    try:
        with torch.no_grad():
            # 这里的 get_embedding 是我们之前让您加到模型类里的那个函数
            embeddings = model.get_embedding(X_sub).cpu().numpy()
    except AttributeError:
        print("   [错误] 模型缺少 'get_embedding' 方法。请确保 step2_train_nn.py 里的类定义已更新。")
        return

    # === [步骤 4] 运行 t-SNE ===
    # Perplexity 设为样本数的 1/5 到 1/10 之间通常效果不错，但这不能超过样本数
    curr_perp = min(30, max(5, int(len(final_idx) / 5)))

    print(f"   -> 开始运行 t-SNE (Perplexity={curr_perp})...")
    tsne = TSNE(n_components=2, perplexity=curr_perp, max_iter=3000, random_state=42, init='pca', learning_rate='auto',early_exaggeration = 40)
    X_embedded = tsne.fit_transform(embeddings)

    # === [步骤 5] 绘图 ===
    # 提取对应的标签用于上色
    labels_sub = labels[final_idx]

    # 颜色映射: 0=红(CV), 1=绿(CA), 2=蓝(CT)
    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    label_map = {0: 'CV', 1: 'CA', 2: 'CT '}

    plot_colors = [color_map[l] for l in labels_sub]

    plt.figure(figsize=(9, 9))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=plot_colors, alpha=0.6, s=23, edgecolors='none')

    # 手动创建图例
    legend_patches = []
    unique_labels_in_plot = np.unique(labels_sub)
    for tag in unique_labels_in_plot:
        patch = mpatches.Patch(color=color_map[tag], label=label_map[tag])
        legend_patches.append(patch)

    plt.legend(handles=legend_patches, title="Flight Mode", loc="upper right")
    plt.title('t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(">>> t-SNE 分析完成。")



# === [优化] 统计相关性矩阵 ===
def analyze_statistical_correlation(X, Y):
    print("\n>>> 3. 正在绘制统计特征相关性热力图...")

    # 计算序列的统计特征
    X_np = X.numpy()  # (N, 90, 9)
    X_mean = np.mean(X_np, axis=1)  # (N, 9) 平均值
    X_std = np.std(X_np, axis=1)  # (N, 9) 标准差

    # 拼接 Mean 和 Std
    # 特征名扩展
    feat_names_ext = [f'{n}_mean' for n in FEATURE_NAMES] + [f'{n}_std' for n in FEATURE_NAMES]
    data_feat = np.hstack([X_mean, X_std])  # (N, 18)

    Y_np = Y.numpy()
    param_names = ['P11', 'P12', 'P21', 'P22', 'P31', 'P32']

    # 计算相关性 (feat vs params)
    full_data = np.hstack([data_feat, Y_np])
    corr_matrix = np.corrcoef(full_data, rowvar=False)

    # 截取 (Feature rows, Param cols)
    # Feat indices: 0~17, Param indices: 18~23
    corr_sub = corr_matrix[:18, 18:]

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=param_names, yticklabels=feat_names_ext)
    plt.title("Correlation: Sequence Statistics (Mean/Std) vs IMM Parameters")
    plt.tight_layout()
    plt.show()

def main():
    setup_seed(414)
    # 1. 准备模型
    model = ParamPredictorMLP().to(DEVICE)
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            print(f"模型 {MODEL_FILE} 加载成功。")
        except:
            print("模型加载失败，将使用随机初始化模型运行（仅演示功能）")
    else:
        print(f"提示：找不到模型文件 {MODEL_FILE}，使用 随机初始化模型。")

    # 2. 准备数据
    X, Y = load_data()
    print(f"数据形状: X={X.shape}, Y={Y.shape}")

    # 3. 运行高级分析
    # (A) 梯度显著性图 - 可以在论文中分析模型的时间敏感性
    analyze_saliency_map(model, X, Y)

    # (B) t-SNE 分布 - 可以在论文中分析模型是否学习到了模式的差异
    analyze_tsne_distribution(model, X, Y)

    # (C) 统计相关性 - 比单纯看最后一帧更有物理意义
    analyze_statistical_correlation(X, Y)


if __name__ == '__main__':
    main()