# 04_build_graph.py
import numpy as np
import json
import os
import torch
from sklearn.model_selection import train_test_split
import time

print("=" * 60)
print("BUILDING GRAPH DATASET")
print("=" * 60)

# 1. 检查环境
try:
    import torch_geometric
    print("✓ PyTorch Geometric available")
except ImportError:
    print("✗ PyTorch Geometric not found")
    print("Install with: pip install torch_geometric")
    exit(1)

# 2. 加载数据
print("\n[1] Loading data...")
data_dir = "processed_data"

# 加载特征矩阵
features = np.load(os.path.join(data_dir, 'features.npy'))
print(f"   Features shape: {features.shape}")

# 加载ID映射
with open(os.path.join(data_dir, 'id_map_processed.json'), 'r') as f:
    id_map = json.load(f)

# 加载类别映射
with open(os.path.join(data_dir, 'class_map_processed.json'), 'r') as f:
    class_map = json.load(f)

# 创建索引到类别的映射
# id_map: {node_id: index}
# class_map: {node_id: class_label}
labels = np.zeros(len(features), dtype=np.int64)
for node_id, idx in id_map.items():
    if node_id in class_map:
        labels[idx] = class_map[node_id]
    else:
        labels[idx] = -1  # 如果没有类别，标记为-1

# 移除没有类别的节点
valid_mask = labels != -1
features = features[valid_mask]
labels = labels[valid_mask]

print(f"   Valid labels shape: {labels.shape}")
print(f"   Unique labels: {len(np.unique(labels))}")

# 3. 选择数据规模
print("\n[2] Select dataset size:")
print(f"   1. Full dataset ({len(features):,} nodes)")
print("   2. Medium subset (50K nodes)")
print("   3. Small subset (10K nodes, for testing)")

choice = input("   Enter choice (1/2/3, default=2): ").strip() or "2"

if choice == "1":
    num_nodes = len(features)
    print(f"   Using full dataset: {num_nodes:,} nodes")
elif choice == "2":
    num_nodes = 50000
    print(f"   Using medium subset: {num_nodes:,} nodes")
    # 随机采样
    indices = np.random.choice(len(features), num_nodes, replace=False)
    features = features[indices]
    labels = labels[indices]
else:
    num_nodes = 10000
    print(f"   Using small subset: {num_nodes:,} nodes")
    indices = np.random.choice(len(features), num_nodes, replace=False)
    features = features[indices]
    labels = labels[indices]

# 4. 构建图结构
print("\n[3] Building graph structure...")

# 检查是否安装了scikit-learn
try:
    from sklearn.neighbors import kneighbors_graph
    print("   Building KNN graph (k=20)...")
    start_time = time.time()
    
    # 使用特征构建KNN图
    k = 20
    adj_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    
    # 转换为边索引
    edges = adj_matrix.nonzero()
    edge_index = np.vstack([edges[0], edges[1]])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    print(f"   Graph built in {time.time()-start_time:.1f}s")
    print(f"   Edges: {edge_index.shape[1]:,}")
    
except ImportError:
    print("   scikit-learn not found, creating random graph...")
    # 创建随机图作为备选
    num_edges = num_nodes * 20  # 每个节点平均20条边
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    print(f"   Random graph created with {num_edges:,} edges")

# 5. 创建PyG数据对象
print("\n[4] Creating PyG data object...")

from torch_geometric.data import Data

x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print(f"   Data object created:")
print(f"     Nodes: {data.num_nodes:,}")
print(f"     Edges: {data.num_edges:,}")
print(f"     Features: {data.num_features}")
print(f"     Classes: {data.y.max().item() + 1}")

# 6. 数据集划分
print("\n[5] Splitting dataset...")

node_indices = list(range(data.num_nodes))
train_idx, temp_idx = train_test_split(
    node_indices, 
    test_size=0.3, 
    stratify=data.y.numpy(),
    random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=data.y[temp_idx].numpy(),
    random_state=42
)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print(f"   Train: {len(train_idx):,} nodes ({len(train_idx)/data.num_nodes*100:.1f}%)")
print(f"   Val:   {len(val_idx):,} nodes ({len(val_idx)/data.num_nodes*100:.1f}%)")
print(f"   Test:  {len(test_idx):,} nodes ({len(test_idx)/data.num_nodes*100:.1f}%)")

# 7. 保存数据
print("\n[6] Saving dataset...")
save_dir = "experiments/data"
os.makedirs(save_dir, exist_ok=True)

# PyG格式
torch.save(data, os.path.join(save_dir, 'reddit_graph.pt'))

# 统计信息
data_stats = {
    'num_nodes': data.num_nodes,
    'num_edges': data.num_edges,
    'num_features': data.num_features,
    'num_classes': int(data.y.max().item() + 1),
    'train_size': len(train_idx),
    'val_size': len(val_idx),
    'test_size': len(test_idx),
    'class_distribution': {str(i): int((data.y == i).sum().item()) 
                          for i in range(int(data.y.max().item() + 1))}
}

with open(os.path.join(save_dir, 'dataset_stats.json'), 'w') as f:
    json.dump(data_stats, f, indent=2)

print(f"   Saved: {save_dir}/reddit_graph.pt")
print(f"   Saved: {save_dir}/dataset_stats.json")

# 8. 可视化图统计
print("\n[7] Creating visualization...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 度分布
degrees = torch.zeros(data.num_nodes, dtype=torch.long)
for i in range(data.num_nodes):
    degrees[i] = (data.edge_index[0] == i).sum().item()

axes[0, 0].hist(degrees.numpy(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Node Degree')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Degree Distribution')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# 类别分布
class_counts = [data_stats['class_distribution'][str(i)] for i in range(data_stats['num_classes'])]
axes[0, 1].bar(range(data_stats['num_classes']), class_counts, alpha=0.7, color='darkorange')
axes[0, 1].set_xlabel('Class ID')
axes[0, 1].set_ylabel('Number of Nodes')
axes[0, 1].set_title('Class Distribution')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# 特征值分布
sample_features = data.x[torch.randperm(data.num_nodes)[:1000]].numpy()
axes[1, 0].hist(sample_features.flatten(), bins=100, alpha=0.7, color='forestgreen', edgecolor='black')
axes[1, 0].set_xlabel('Feature Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Feature Value Distribution')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 数据集划分
split_sizes = [data_stats['train_size'], data_stats['val_size'], data_stats['test_size']]
split_labels = ['Train', 'Validation', 'Test']
colors = ['lightblue', 'lightgreen', 'lightcoral']

axes[1, 1].pie(split_sizes, labels=split_labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Dataset Split')

plt.suptitle(f'Reddit Graph Dataset Statistics (n={data.num_nodes:,})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/visualizations/graph_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Visualization saved: experiments/visualizations/graph_statistics.png")

print("\n" + "=" * 60)
print("GRAPH DATASET BUILT SUCCESSFULLY")
print("=" * 60)