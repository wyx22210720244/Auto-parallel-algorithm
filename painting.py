import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 集群内卡型号分布
data_a = [2416, 1584, 704, 384]
labels_a = ["A100", "H800", "PPU-ZW810", "A800"]
data_b = [3088, 2232, 320, 368, 704]
labels_b = ["GeForce-RTX-2080-Ti", "A800", "Tesla-V100", "PPU-ZW810", "GeForce-RTX-3090"]
colors_a = ['lightcoral', 'springgreen', 'turquoise', 'darkorange', "plum"]
colors_b = ["cyan", 'darkorange', "gold", "turquoise", "lightskyblue"]
size_a = [round(x / sum(data_a) * 100) for x in data_a]
size_b = [round(x / sum(data_b) * 100) for x in data_b]
size_b[1] += 1
fig, axs = plt.subplots(2, 1, figsize=(12, 3), dpi=600)
base_a = 0
base_b = 0
for segment, color, label in zip(size_a, colors_a, labels_a):
    axs[0].barh("Cluster A", segment, left=base_a, color=color, label=label)
    axs[0].text(base_a + segment / 2, "Cluster A", f"{segment}%", va='center', ha='center', color='black')
    base_a += segment
axs[0].text(-5, 0, "Cluster A:", ha="center", weight='bold', fontsize=12)
for segment, color, label in zip(size_b, colors_b, labels_b):
    axs[1].barh("Cluster B", segment, left=base_b, color=color, label=label)
    axs[1].text(base_b + segment / 2, "Cluster B", f"{segment}%", va='center', ha='center', color='black')
    base_b += segment
axs[1].text(-5, 0, "Cluster B:", ha="center", weight='bold', fontsize=12)
# 去掉坐标轴
axs[0].axis("off")
axs[1].axis("off")
# 添加图例
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

plt.tight_layout()
plt.show()

# 各型号卡在线分配率的时间线
data = pd.DataFrame({
    "dates": ["0", "8", "16", "24", "32", "40"],
    "A800": [97.3, 97.1, 98.1, 98.3, 98.2, 97.9],
    "GeForce-RTX-2080-Ti": [96.6, 96.3, 96.1, 95.9, 95.8, 96.2],
    "Tesla-V100": [85.7, 86.8, 86.5, 87.1, 86.8, 86.9],
    "GeForce-RTX-3090": [90.1, 92.6, 91.6, 91.7, 91.3, 91.5],
    "A100": [97.7, 97.5, 97.9, 98.1, 97.9, 98.2],
    "H800": [92.1, 91.2, 90.9, 89.5, 89.8, 90.5]
})
fig2, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=600)
for column in data.columns[1:]:
    ax.plot(data["dates"], data[column], marker='o', label=column)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Online allocation ratio (%)")
# ax.set_title("Online rate of each card model over time")
ax.legend(loc=(0.25,-0.5), shadow=True, ncol=3)
plt.tight_layout()
plt.show()