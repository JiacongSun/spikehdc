import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

def percentage(x, pos):
    return f'{100 * x:.0f}'  # Format as percentage



color_list = ['purple','blue','orange','orange','green', 'green','red','red']
linestyle_list = ['-','-','-','--','-','--','-','--']
df1 = pd.read_csv('response/all_accuracy_merged_transposed_reordered.csv')
header = df1.columns.tolist()
header_ref = ['AT[1]','DVT[4]','ED[2]','ED-DVT','Δ[3]','Δ-DVT','NEO[2]','NEO-DVT']
all_accuracies = df1.values.tolist()
transposed_accuracies = list(zip(*all_accuracies))


fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))  # 一行两列的子图，整体宽度14，高度6


for i, acc in enumerate(transposed_accuracies):
    axes[0].plot( np.arange(1, 1 + 0.05 * len(acc), 0.05),acc,linestyle=linestyle_list[i],color=color_list[i], label=f"{header_ref[i]}")


# Add labels, title, and legend
#axes[0].set_title("Accuracy in Different Methods")
axes[0].set_xlabel("Threshold Factor $k$")
axes[0].set_ylabel("Accuracy(%)")
axes[0].legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True)
axes[0].grid(True)


# 假设的最高精度数据（可以替换成你自己的数据）
# 每个方法和emphasizer对应的最高精度
max_accuracies = [max(acc) for acc in transposed_accuracies]


# 设置柱形图参数
x = np.arange(len(header))  # 每个emphasizer的x轴位置
width = 0.6  # 每个柱子的宽度

# 创建柱形图

bars = axes[1].bar(x , max_accuracies, width,  color=color_list, linewidth=0)  # 深蓝色

for i in [3, 5, 7]:  # 4th, 6th, 8th bars have index 3, 5, and 7 respectively
    axes[1].bar(i, max_accuracies, color='none', edgecolor='white', hatch='//', linewidth=0)  # 白色斜线
# 在柱子上方添加最高精度值
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()*100:.2f}', 
             ha='center', va='bottom', fontsize=10,rotation=30)
 #添加标签和标题
axes[1].set_xlabel('Methods')
axes[1].set_ylabel('Highest Accuracy(%)')
#axes[1].set_title('Highest Accuracy in Different Methods')
axes[1].set_xticks(x, header,rotation=30, ha='center')  # 设置x轴标签
axes[1].set_ylim(0.85, 1)  # 设置纵轴范围

axes[0].yaxis.set_major_formatter(FuncFormatter(percentage))
axes[1].yaxis.set_major_formatter(FuncFormatter(percentage))
# 显示图形
plt.tight_layout()

# Save the plot
plt.savefig("accuracy_curves.png")
plt.show()