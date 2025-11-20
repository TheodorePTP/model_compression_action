import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

# 创建保存目录
save_dir = "./assets/distillation/diagrams"
os.makedirs(save_dir, exist_ok=True)

# 设置中文字体和样式
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def softmax(logits, temperature=1.0):
    """计算带温度系数的softmax"""
    logits = np.array(logits) / temperature
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
    return exp_logits / np.sum(exp_logits)

def plot_temperature_effect():
    # 示例logits（5个类别的原始输出）
    logits = [2.0, 1.0, 0.1, -1.0, -2.0]
    classes = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
    
    # 不同的温度值
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：不同温度下的概率分布
    x = np.arange(len(classes))
    width = 0.15
    
    for i, temp in enumerate(temperatures):
        probs = softmax(logits, temp)
        ax1.bar(x + i * width, probs, width, label=f'T={temp}', 
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Temperature on Softmax Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(classes)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 右图：温度对最大概率和熵的影响
    temp_range = np.linspace(0.1, 15, 100)
    max_probs = []
    entropies = []
    
    for temp in temp_range:
        probs = softmax(logits, temp)
        max_probs.append(np.max(probs))
        # 计算熵
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(entropy)
    
    ax2.plot(temp_range, max_probs, 'b-', linewidth=2.5, label='Max Probability')
    ax2.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Probability', fontsize=12, fontweight='bold', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(temp_range, entropies, 'r-', linewidth=2.5, label='Entropy')
    ax2_twin.set_ylabel('Entropy', fontsize=12, fontweight='bold', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # 添加标记点
    marker_temps = [0.5, 1.0, 2.0, 5.0, 10.0]
    for temp in marker_temps:
        probs = softmax(logits, temp)
        max_prob = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        ax2.plot(temp, max_prob, 'bo', markersize=8)
        ax2_twin.plot(temp, entropy, 'ro', markersize=8)
    
    ax2.set_title('Temperature vs Max Probability & Entropy', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # 组合图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'{save_dir}/temperature_effect.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{save_dir}/temperature_effect.pdf', bbox_inches='tight')
    
    plt.show()
    
    # 打印具体数值表格
    print("\n不同温度下的概率分布:")
    print("Temperature |", " | ".join(classes))
    print("-" * 60)
    for temp in temperatures:
        probs = softmax(logits, temp)
        prob_str = " | ".join([f"{p:.4f}" for p in probs])
        print(f"T={temp:<4}     | {prob_str}")

def plot_temperature_comparison():
    """绘制温度对比图"""
    logits = [3.0, 2.0, 1.0, 0.0, -1.0]
    classes = ['Cat', 'Dog', 'Bird', 'Fish', 'Rabbit']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    titles = ['Very Low (T=0.1)', 'Low (T=0.5)', 'Normal (T=1.0)', 
              'Medium (T=2.0)', 'High (T=5.0)', 'Very High (T=10.0)']
    
    for i, (temp, title) in enumerate(zip(temperatures, titles)):
        probs = softmax(logits, temp)
        
        # 柱状图
        bars = axes[i].bar(classes, probs, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[i].set_title(f'{title}\nEntropy: { -np.sum(probs * np.log(probs + 1e-8)):.3f}', 
                         fontweight='bold')
        axes[i].set_ylabel('Probability')
        axes[i].tick_params(axis='x', rotation=45)
        
        # 在柱子上添加数值
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/temperature_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/temperature_comparison.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("生成温度系数影响图...")
    plot_temperature_effect()
    print(f"图片已保存到: {save_dir}/temperature_effect.png")
    
    print("\n生成温度对比图...")
    plot_temperature_comparison()
    print(f"图片已保存到: {save_dir}/temperature_comparison.png")