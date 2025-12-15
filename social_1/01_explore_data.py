# 01_explore_data.py
import numpy as np
import json
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from collections import Counter

print("=" * 60)
print("Reddit数据集探索")
print("=" * 60)

# 1. 检查文件
data_dir = "reddit"
files = os.listdir(data_dir)
print("数据集文件列表:")
for file in files:
    size = os.path.getsize(os.path.join(data_dir, file))
    print(f"  {file}: {size:,} bytes")

print("\n" + "-" * 60)

# 2. 加载类别映射
print("1. 加载类别映射 (reddit-class_map.json)...")
with open(os.path.join(data_dir, "reddit-class_map.json"), 'r') as f:
    class_map = json.load(f)
print(f"   类别数量: {len(class_map)}")
print(f"   示例 (前5个): {list(class_map.items())[:5]}")

# 检查类别分布
class_counter = Counter(class_map.values())
print(f"   类别分布: {dict(class_counter)}")

# 3. 加载ID映射
print("\n2. 加载ID映射 (reddit-id_map.json)...")
with open(os.path.join(data_dir, "reddit-id_map.json"), 'r') as f:
    id_map = json.load(f)
print(f"   节点数量: {len(id_map)}")
print(f"   示例 (前5个): {list(id_map.items())[:5]}")

# 4. 加载特征矩阵
print("\n3. 加载特征矩阵 (reddit-feats.npy)...")
feats = np.load(os.path.join(data_dir, "reddit-feats.npy"))
print(f"   特征矩阵形状: {feats.shape}")
print(f"   特征维度: {feats.shape[1]}")
print(f"   数据类型: {feats.dtype}")
print(f"   稀疏度: {100 * (feats == 0).sum() / feats.size:.2f}%")

# 5. 加载邻接列表
print("\n4. 加载邻接列表 (reddit-adjlist.txt)...")
adj_list = {}
with open(os.path.join(data_dir, "reddit-adjlist.txt"), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            node = int(parts[0])
            neighbors = list(map(int, parts[1:]))
            adj_list[node] = neighbors

print(f"   节点数量 (有邻接关系的): {len(adj_list)}")

# 计算度分布
degrees = [len(neighbors) for neighbors in adj_list.values()]
print(f"   平均度: {np.mean(degrees):.2f}")
print(f"   最大度: {max(degrees)}")
print(f"   最小度: {min(degrees)}")

# 6. 加载图结构 (可选)
print("\n5. 加载图结构 (reddit-G.json)...")
try:
    with open(os.path.join(data_dir, "reddit-G.json"), 'r') as f:
        G_data = json.load(f)
    print(f"   图节点数: {len(G_data.get('nodes', []))}")
    print(f"   图边数: {len(G_data.get('links', []))}")
except Exception as e:
    print(f"   加载失败: {e}")

# 7. 加载随机游走 (如果存在)
print("\n6. 检查随机游走文件 (reddit-walks.txt)...")
if os.path.exists(os.path.join(data_dir, "reddit-walks.txt")):
    with open(os.path.join(data_dir, "reddit-walks.txt"), 'r') as f:
        first_line = f.readline().strip()
    print(f"   示例游走: {first_line[:100]}...")
    print(f"   游走长度: {len(first_line.split())}")
else:
    print("   随机游走文件不存在")

print("\n" + "=" * 60)
print("数据探索完成！")
print("=" * 60)

# 8. 可视化一些统计数据
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 类别分布
ax1 = axes[0, 0]
class_counts = list(class_counter.values())
ax1.hist(class_counts, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('每个类别的节点数')
ax1.set_ylabel('类别数量')
ax1.set_title('类别分布')
ax1.set_yscale('log')

# 度分布
ax2 = axes[0, 1]
ax2.hist(degrees, bins=50, range=(0, 100), edgecolor='black', alpha=0.7)
ax2.set_xlabel('节点度')
ax2.set_ylabel('节点数量')
ax2.set_title('度分布')
ax2.set_yscale('log')

# 特征稀疏性
ax3 = axes[1, 0]
sample_node = np.random.randint(0, feats.shape[0])
sample_features = feats[sample_node]
non_zero_idx = np.where(sample_features != 0)[0]
ax3.stem(non_zero_idx[:50], sample_features[non_zero_idx[:50]])
ax3.set_xlabel('特征维度')
ax3.set_ylabel('特征值')
ax3.set_title(f'节点 {sample_node} 的特征稀疏模式 (前50个非零)')

# 节点类别示例
ax4 = axes[1, 1]
class_ids = list(class_counter.keys())
class_counts = list(class_counter.values())
top_n = min(10, len(class_ids))
top_classes = sorted(zip(class_counts, class_ids), reverse=True)[:top_n]
ax4.bar(range(top_n), [count for count, _ in top_classes])
ax4.set_xlabel('类别ID')
ax4.set_ylabel('节点数量')
ax4.set_title('Top 10 类别')
ax4.set_xticks(range(top_n))
ax4.set_xticklabels([cid for _, cid in top_classes], rotation=45)

plt.tight_layout()
plt.savefig('01_data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 保存数据统计信息
stats = {
    "num_nodes": feats.shape[0],
    "num_features": feats.shape[1],
    "num_classes": len(class_counter),
    "avg_degree": np.mean(degrees),
    "max_degree": max(degrees),
    "min_degree": min(degrees),
    "feature_sparsity": 100 * (feats == 0).sum() / feats.size,
    "class_distribution": dict(class_counter)
}

with open('01_data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n数据统计已保存到 '01_data_stats.json'")
print("可视化图表已保存到 '01_data_exploration.png'")