# 03_study_design.py
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# 设置matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("STUDY DESIGN: User Interest Profiling with GNN")
print("=" * 60)

# 1. 加载数据统计
with open('processed_data/data_statistics.json', 'r') as f:
    stats = json.load(f)

print(f"Nodes: {stats['num_nodes']:,}")
print(f"Features: {stats['num_features']}")
print(f"Classes: {stats['num_classes']}")
print(f"Feature sparsity: {stats['feature_sparsity']}")

# 2. 创建实验目录
experiment_dirs = [
    "experiments",
    "experiments/data",
    "experiments/models",
    "experiments/results",
    "experiments/visualizations"
]

for dir_path in experiment_dirs:
    os.makedirs(dir_path, exist_ok=True)

print(f"\nDirectories created: {experiment_dirs}")

# 3. 保存研究方案
study_plan = {
    "title": "Graph Neural Networks for Social Network User Profiling",
    "dataset": "Reddit",
    "num_nodes": stats['num_nodes'],
    "num_features": stats['num_features'],
    "num_classes": stats['num_classes'],
    "objectives": [
        "Build user profiles combining features and social relations",
        "Accurate user interest classification",
        "Analyze social influence on user interests"
    ],
    "methodology": "Graph Convolutional Networks with attention mechanism",
    "evaluation": ["Accuracy", "F1-score", "AUC", "NMI"]
}

with open('experiments/study_plan.json', 'w', encoding='utf-8') as f:
    json.dump(study_plan, f, indent=2)

print(f"\nStudy plan saved: experiments/study_plan.json")

# 4. 可视化类别分布
class_dist = stats['class_distribution']
class_ids = list(class_dist.keys())
class_counts = list(class_dist.values())

# 排序
sorted_indices = np.argsort(class_counts)[::-1]
class_ids_sorted = [class_ids[i] for i in sorted_indices]
class_counts_sorted = [class_counts[i] for i in sorted_indices]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 条形图
ax1.bar(range(len(class_ids_sorted)), class_counts_sorted, alpha=0.7, color='steelblue')
ax1.set_xlabel('Class ID')
ax1.set_ylabel('Number of Users')
ax1.set_title('User Interest Class Distribution')
ax1.set_xticks(range(len(class_ids_sorted)))
ax1.set_xticklabels(class_ids_sorted, rotation=90, fontsize=8)
ax1.grid(True, alpha=0.3)

# 饼图 (Top 10)
top_n = 10
other_count = sum(class_counts_sorted[top_n:])
pie_labels = class_ids_sorted[:top_n] + ['Other']
pie_sizes = class_counts_sorted[:top_n] + [other_count]

ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, 
        colors=plt.cm.Set3(np.linspace(0, 1, top_n + 1)))
ax2.set_title(f'Top {top_n} Interest Classes')

plt.suptitle('Reddit User Interest Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved: experiments/visualizations/class_distribution.png")
print("\n" + "=" * 60)
print("STUDY DESIGN COMPLETED")
print("=" * 60)