# 02_preprocess_data.py
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import time

# 设置matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Reddit Dataset Preprocessing")
print("=" * 60)

data_dir = "reddit"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# 1. 重新加载类别映射和ID映射
print("1. Loading mapping files...")
with open(os.path.join(data_dir, "reddit-class_map.json"), 'r') as f:
    class_map = json.load(f)

with open(os.path.join(data_dir, "reddit-id_map.json"), 'r') as f:
    id_map = json.load(f)

# 反转ID映射：索引->原始ID
reverse_id_map = {v: k for k, v in id_map.items()}

print(f"   Total nodes: {len(id_map)}")
print(f"   Number of classes: {len(set(class_map.values()))}")

# 2. 加载特征矩阵
print("\n2. Loading feature matrix...")
feats = np.load(os.path.join(data_dir, "reddit-feats.npy"))
print(f"   Feature matrix shape: {feats.shape}")

# 3. 修正邻接列表加载
print("\n3. Loading adjacency list (skipping comment lines)...")

def load_adjacency_list(file_path, max_lines=None):
    """加载邻接列表，跳过注释行"""
    adjacency_dict = {}
    line_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过空行和注释行
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            parts = line.strip().split()
            if parts:
                # 第一个部分是字符串ID
                node_str = parts[0]
                
                # 转换为整数索引
                if node_str in id_map:
                    node_idx = id_map[node_str]
                    
                    # 转换邻居
                    neighbors = []
                    for neighbor_str in parts[1:]:
                        if neighbor_str in id_map:
                            neighbors.append(id_map[neighbor_str])
                    
                    adjacency_dict[node_idx] = neighbors
                    
                    line_count += 1
                    
                    # 如果设置了最大行数，检查是否达到
                    if max_lines and line_count >= max_lines:
                        break
    
    return adjacency_dict

# 先加载部分数据测试（比如前1000个节点）
adj_list_partial = load_adjacency_list(
    os.path.join(data_dir, "reddit-adjlist.txt"), 
    max_lines=1000
)
print(f"   Loaded adjacency relations for {len(adj_list_partial)} nodes")

# 4. 分析类别分布
print("\n4. Analyzing class distribution...")
class_values = list(class_map.values())
class_counter = Counter(class_values)

print(f"   Total classes: {len(class_counter)}")
print(f"   Class distribution statistics:")

# 按节点数排序
sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
for i, (class_id, count) in enumerate(sorted_classes[:10]):
    print(f"     Class {class_id}: {count} nodes ({count/len(class_map)*100:.2f}%)")

# 5. 创建用户画像任务的数据结构
print("\n5. Creating user profile dataset...")

# 为每个用户（节点）创建特征向量
def create_user_features():
    """创建用户特征数据集"""
    num_users = feats.shape[0]
    
    # 1. 静态特征（原始特征矩阵）
    print(f"   - Static features: {num_users} users, {feats.shape[1]} dimensions")
    
    # 2. 动态特征（基于邻居关系）
    print("   - Building dynamic features (based on graph structure)...")
    
    # 简单统计：每个用户的邻居类别分布
    user_neighbor_classes = {}
    
    # 为每个用户统计邻居的类别
    for user_idx in range(min(1000, num_users)):  # 先处理1000个用户测试
        if user_idx in adj_list_partial:
            neighbors = adj_list_partial[user_idx]
            
            # 统计邻居类别
            neighbor_classes = []
            for neighbor_idx in neighbors[:10]:  # 最多取10个邻居
                neighbor_id = reverse_id_map.get(neighbor_idx)
                if neighbor_id and neighbor_id in class_map:
                    neighbor_classes.append(class_map[neighbor_id])
            
            if neighbor_classes:
                # 计算类别分布
                class_counts = Counter(neighbor_classes)
                user_neighbor_classes[user_idx] = dict(class_counts)
    
    print(f"   - Neighbor class distribution calculated for {len(user_neighbor_classes)} users")
    
    return user_neighbor_classes

user_neighbor_classes = create_user_features()

# 6. 保存预处理后的数据
print("\n6. Saving preprocessed data...")

# 保存类别映射
with open(os.path.join(output_dir, 'class_map_processed.json'), 'w') as f:
    json.dump(class_map, f)

# 保存ID映射
with open(os.path.join(output_dir, 'id_map_processed.json'), 'w') as f:
    json.dump(id_map, f)

# 保存特征矩阵（可以保存为.npy或稀疏格式）
np.save(os.path.join(output_dir, 'features.npy'), feats)

# 保存用户邻居类别信息
with open(os.path.join(output_dir, 'user_neighbor_classes.json'), 'w') as f:
    json.dump(user_neighbor_classes, f)

# 7. 可视化
print("\n7. Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 类别分布直方图
ax1 = axes[0, 0]
class_counts = [count for _, count in sorted_classes]
ax1.hist(class_counts, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of nodes per class')
ax1.set_ylabel('Number of classes')
ax1.set_title('Class Distribution')
ax1.set_yscale('log')

# Top 20类别
ax2 = axes[0, 1]
top_n = min(20, len(sorted_classes))
class_ids = [str(cid) for cid, _ in sorted_classes[:top_n]]
counts = [count for _, count in sorted_classes[:top_n]]
bars = ax2.barh(range(top_n), counts)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(class_ids)
ax2.invert_yaxis()
ax2.set_xlabel('Number of nodes')
ax2.set_title(f'Top {top_n} Most Frequent Classes')

# 特征稀疏性
ax3 = axes[1, 0]
sample_indices = np.random.choice(feats.shape[0], 100, replace=False)
sparsity_per_user = [(feats[i] == 0).sum() / feats.shape[1] for i in sample_indices]
ax3.hist(sparsity_per_user, bins=20, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Feature sparsity (zero ratio)')
ax3.set_ylabel('Number of users')
ax3.set_title('Feature Sparsity Distribution (100 random users)')

# 度分布（基于部分数据）
ax4 = axes[1, 1]
if adj_list_partial:
    degrees = [len(neighbors) for neighbors in adj_list_partial.values()]
    ax4.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Node degree (number of neighbors)')
    ax4.set_ylabel('Number of nodes')
    ax4.set_title(f'Degree Distribution ({len(adj_list_partial)} nodes)')
    ax4.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_preprocessing_visualization.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Preprocessing completed!")
print(f"Processed data saved in: {output_dir}/")
print("=" * 60)

# 8. 生成数据报告
data_stats = {
    "dataset": "Reddit",
    "num_nodes": feats.shape[0],
    "num_features": feats.shape[1],
    "num_classes": len(class_counter),
    "class_distribution": dict(sorted_classes),
    "feature_sparsity": f"{100*(feats==0).sum()/feats.size:.2f}%",
    "sample_users_processed": len(user_neighbor_classes),
    "preprocessing_date": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(output_dir, 'data_statistics.json'), 'w') as f:
    json.dump(data_stats, f, indent=2)

print(f"\nData statistics saved to: {output_dir}/data_statistics.json")