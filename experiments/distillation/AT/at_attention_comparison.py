```
脚本用途：
https://blog.csdn.net/qq_44923064/article/details/155104865?fromshare=blogdetail&sharetype=blogdetail&sharerId=155104865&sharerefer=PC&sharesource=qq_44923064&sharefrom=from_link
上文图3和图4图5的生成脚本

使用方法：
python at_attention_comparison.py

```

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

# ----------------------------
# 1. 模拟教师和学生特征图（简化版）
# ----------------------------
def get_dummy_features(batch=1, c_t=512, c_s=256, h=14, w=14):
    """生成模拟的教师和学生特征图"""
    torch.manual_seed(42)
    f_t = torch.randn(batch, c_t, h, w) * 0.5 + 1.0  # 教师特征（更“聚焦”）
    f_s = torch.randn(batch, c_s, h, w) * 1.0        # 学生特征（更“分散”）
    return f_s, f_t

def compute_attention_map(feat, p=2):
    """计算 AT 论文中的注意力图: sum of power-p across channels"""
    att = feat.pow(p).mean(dim=1)  # [B, H, W]
    # Normalize each attention map independently, without flattening it
    att_normalized = F.normalize(att.view(att.size(0), -1), p=2, dim=1).view_as(att)
    return att_normalized  # keep [B, H, W]


# ----------------------------
# 图1：教师 vs 学生注意力图
# ----------------------------
def plot_attention_comparison():
    # 假设 get_dummy_features 是一个已定义的函数，它返回学生和教师模型的特征
    # 如果有真实的特征，请直接使用它们代替这个假设函数
    def get_dummy_features(h=28, w=28):
        # 示例数据，实际中应由模型生成
        f_s = torch.randn((1, 64, h, w))  # 学生模型特征
        f_t = torch.randn((1, 64, h, w))  # 教师模型特征
        return f_s, f_t
    
    f_s, f_t = get_dummy_features(h=28, w=28)
    att_s = compute_attention_map(f_s, p=2)[0].detach().numpy()  # Ensure it's a single map and 2D
    att_t = compute_attention_map(f_t, p=2)[0].detach().numpy()  # Ensure it's a single map and 2D

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    im0 = axs[0].imshow(att_t, cmap='jet', interpolation='bilinear')
    axs[0].set_title('Teacher Attention Map', fontsize=12)
    axs[0].axis('off')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(att_s, cmap='jet', interpolation='bilinear')
    axs[1].set_title('Student Attention Map (before AT)', fontsize=12)
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('at_attention_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: at_attention_comparison.png")
# ----------------------------
# 图2：AT 蒸馏流程图
# ----------------------------
def plot_at_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # 教师分支
    ax.text(0.2, 0.7, 'Teacher Network', ha='center', fontsize=12, weight='bold')
    ax.add_patch(plt.Rectangle((0.05, 0.55), 0.3, 0.1, fill=None, edgecolor='blue'))
    ax.text(0.2, 0.6, 'Feature Map $F_t$', ha='center', fontsize=11)

    # 学生分支
    ax.text(0.2, 0.3, 'Student Network', ha='center', fontsize=12, weight='bold')
    ax.add_patch(plt.Rectangle((0.05, 0.15), 0.3, 0.1, fill=None, edgecolor='orange'))
    ax.text(0.2, 0.2, 'Feature Map $F_s$', ha='center', fontsize=11)

    # 注意力计算
    ax.annotate('', xy=(0.45, 0.6), xytext=(0.35, 0.6), arrowprops=dict(arrowstyle='->', color='blue'))
    ax.text(0.5, 0.6, 'Attention\n$A_t = \\|F_t\\|_p^p$', ha='center', fontsize=11, color='blue')

    ax.annotate('', xy=(0.45, 0.2), xytext=(0.35, 0.2), arrowprops=dict(arrowstyle='->', color='orange'))
    ax.text(0.5, 0.2, 'Attention\n$A_s = \\|F_s\\|_p^p$', ha='center', fontsize=11, color='orange')

    # 损失函数
    ax.annotate('', xy=(0.7, 0.4), xytext=(0.6, 0.6), arrowprops=dict(arrowstyle='->', color='blue'))
    ax.annotate('', xy=(0.7, 0.4), xytext=(0.6, 0.2), arrowprops=dict(arrowstyle='->', color='orange'))
    ax.text(0.75, 0.4, '$\\mathcal{L}_{AT} = \\|A_t - A_s\\|_2^2$', ha='center', fontsize=12, weight='bold')

    plt.title('Attention Transfer (AT) Knowledge Distillation', fontsize=14, y=0.95)
    plt.savefig('at_pipeline.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: at_pipeline.png")

# ----------------------------
# 图3：特征图 → 注意力图转换
# ----------------------------
def plot_feature_to_attention():
    # 模拟一个 3x3 特征图（3通道）
    np.random.seed(0)
    feat = np.random.rand(3, 32, 32)
    feat = (feat - feat.min()) / (feat.max() - feat.min())  # normalize to [0,1]

    # 计算注意力：sum of squares across channels
    attention = np.sum(feat ** 2, axis=0)

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(3):
        axs[0, i].imshow(feat[i], cmap='viridis')
        axs[0, i].set_title(f'Channel {i+1}', fontsize=10)
        axs[0, i].axis('off')
    
    axs[0, 3].axis('off')  # 空白

    # 第二行：注意力图
    for i in range(3):
        axs[1, i].axis('off')
    im = axs[1, 1].imshow(attention, cmap='hot')
    axs[1, 1].set_title('Sum of Squares\n(Across Channels)', fontsize=10)
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    axs[1, 0].text(0.5, 0.5, '→', fontsize=20, ha='center', va='center')
    axs[1, 0].axis('off')
    axs[1, 2].axis('off')
    axs[1, 3].axis('off')

    plt.suptitle('From Multi-channel Features to Spatial Attention Map', fontsize=13)
    plt.tight_layout()
    plt.savefig('at_feature_to_attention.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: at_feature_to_attention.png")

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    plot_attention_comparison()
    plot_at_pipeline()
    plot_feature_to_attention()
    print("\n🎉 All figures generated successfully!")