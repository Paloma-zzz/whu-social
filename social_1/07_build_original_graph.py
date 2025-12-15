# 07_build_original_graph.py
import numpy as np
import json
import os
import torch
from collections import defaultdict
import time

print("=" * 60)
print("BUILDING ORIGINAL SOCIAL GRAPH")
print("=" * 60)

# 1. 加载数据
print("\n[1] Loading data...")
features = np.load('processed_data/features.npy')

with open('processed_data/id_map_processed.json', 'r') as f:
    id_map = json.load(f)

with open('processed_data/class_map_processed.json', 'r') as f:
    class_map = json.load(f)

# 2. 选择适中规模
sample_size = 30000  # 比之前小，但用真实图
print(f"   Selecting {sample_size:,} nodes...")

# 选择活跃节点（基于特征非零数）
nonzero_counts = (features != 0).sum(axis=1)
top_indices = np.argsort(-nonzero_counts)[:sample_size]  # 选择特征最丰富的节点

features_sampled = features[top_indices]

# 创建映射
original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(top_indices)}
reverse_id_map = {v: k for k, v in id_map.items()}

# 3. 加载原始邻接关系
print("\n[2] Loading original adjacency relations...")

adj_list = defaultdict(list)
processed = 0
start_time = time.time()

with open('reddit/reddit-adjlist.txt', 'r') as f:
    for line in f:
        if processed % 100000 == 0 and processed > 0:
            elapsed = time.time() - start_time
            print(f"   Processed {processed:,} lines, found {len(adj_list)} edges, time: {elapsed:.1f}s")
        
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        parts = line.strip().split()
        if len(parts) >= 2:
            src_str = parts[0]
            
            # 检查是否在采样节点中
            if src_str in id_map and id_map[src_str] in original_to_new:
                src_new = original_to_new[id_map[src_str]]
                
                # 添加邻居
                for neighbor_str in parts[1:]:
                    if neighbor_str in id_map:
                        neighbor_orig = id_map[neighbor_str]
                        if neighbor_orig in original_to_new:
                            dst_new = original_to_new[neighbor_orig]
                            if src_new != dst_new:  # 避免自环
                                adj_list[src_new].append(dst_new)
        
        processed += 1
        if processed >= 2000000:  # 限制处理行数
            break

print(f"   Total edges found: {sum(len(v) for v in adj_list.values()):,}")

# 4. 构建边索引
print("\n[3] Building edge index...")
edge_list = []
for src, neighbors in adj_list.items():
    for dst in neighbors:
        edge_list.append([src, dst])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print(f"   Edge index shape: {edge_index.shape}")

# 5. 创建标签
print("\n[4] Creating labels...")
labels = np.zeros(sample_size, dtype=np.int64)
for new_idx, orig_idx in enumerate(top_indices):
    node_id = reverse_id_map.get(orig_idx)
    if node_id and node_id in class_map:
        labels[new_idx] = class_map[node_id]

# 移除没有标签的节点
valid_mask = labels != 0
features_final = features_sampled[valid_mask]
labels_final = labels[valid_mask]

# 重新映射边
valid_indices = np.where(valid_mask)[0]
valid_map = {old: new for new, old in enumerate(valid_indices)}

filtered_edges = []
for i in range(edge_index.shape[1]):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    if src in valid_map and dst in valid_map:
        filtered_edges.append([valid_map[src], valid_map[dst]])

if filtered_edges:
    edge_index_final = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
else:
    # 如果过滤后没有边，创建一些随机边
    print("   Warning: No edges after filtering, creating random edges")
    num_nodes = len(features_final)
    edge_list = []
    for i in range(num_nodes):
        for j in range(i+1, min(i+10, num_nodes)):
            edge_list.append([i, j])
            edge_list.append([j, i])
    edge_index_final = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"   Final graph: {len(features_final):,} nodes, {edge_index_final.shape[1]:,} edges")

# 6. 创建PyG数据对象
print("\n[5] Creating PyG data object...")
from torch_geometric.data import Data

data = Data(
    x=torch.tensor(features_final, dtype=torch.float32),
    edge_index=edge_index_final,
    y=torch.tensor(labels_final, dtype=torch.long)
)

# 7. 划分数据集
print("\n[6] Splitting dataset...")
from sklearn.model_selection import train_test_split

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

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

print(f"   Train: {len(train_idx):,} nodes ({len(train_idx)/data.num_nodes*100:.1f}%)")
print(f"   Val: {len(val_idx):,} nodes ({len(val_idx)/data.num_nodes*100:.1f}%)")
print(f"   Test: {len(test_idx):,} nodes ({len(test_idx)/data.num_nodes*100:.1f}%)")

# 8. 保存数据
print("\n[7] Saving dataset...")
os.makedirs('experiments/data', exist_ok=True)
torch.save(data, 'experiments/data/original_social_graph.pt')

# 统计信息
stats = {
    'num_nodes': data.num_nodes,
    'num_edges': data.num_edges,
    'num_features': data.num_features,
    'num_classes': int(data.y.max().item() + 1),
    'train_size': len(train_idx),
    'val_size': len(val_idx),
    'test_size': len(test_idx),
    'avg_degree': edge_index_final.shape[1] / data.num_nodes,
}

with open('experiments/data/original_graph_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"   Saved: experiments/data/original_social_graph.pt")
print(f"   Stats: {stats}")

print("\n" + "=" * 60)
print("ORIGINAL SOCIAL GRAPH BUILT SUCCESSFULLY")
print("=" * 60)