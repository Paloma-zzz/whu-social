# 08_train_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("TRAINING GAT (GRAPH ATTENTION NETWORK)")
print("=" * 60)

# 1. 加载原始社交图数据
print("\n[1] Loading original social graph...")
data_path = 'experiments/data/original_social_graph.pt'

try:
    data = torch.load(data_path, weights_only=False)
except:
    print(f"   Error loading {data_path}")
    print("   Please run 07_build_original_graph.py first")
    exit(1)

print(f"   Graph loaded:")
print(f"     Nodes: {data.num_nodes:,}")
print(f"     Edges: {data.num_edges:,}")
print(f"     Features: {data.num_features}")
print(f"     Classes: {data.y.max().item() + 1}")

# 特征标准化
print("   Standardizing features...")
mean = data.x.mean(dim=0, keepdim=True)
std = data.x.std(dim=0, keepdim=True) + 1e-8
data.x = (data.x - mean) / std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")
data = data.to(device)

# 2. 构建GAT模型
print("\n[2] Building GAT model...")

class EnhancedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        
        # 第一层：多头注意力
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        # 第二层：单头注意力
        self.conv2 = GATConv(
            hidden_channels * heads, 
            out_channels, 
            heads=1, 
            dropout=dropout,
            concat=False
        )
        
        # 跳跃连接
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
        # 批量归一化
        self.batch_norm = nn.BatchNorm1d(hidden_channels * heads)
        
        # Dropout
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # 保存输入用于跳跃连接
        x_in = x
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层
        x = self.conv2(x, edge_index)
        
        # 跳跃连接
        if self.skip is not None:
            x_skip = self.skip(x_in)
            x = x + x_skip
        
        return F.log_softmax(x, dim=1)

in_channels = data.num_features
hidden_channels = 64  # 每个头的维度
out_channels = data.y.max().item() + 1
heads = 4

model = EnhancedGAT(in_channels, hidden_channels, out_channels, heads=heads, dropout=0.5)
model = model.to(device)

print(f"   Model: GAT with {heads} attention heads")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. 高级训练策略
print("\n[3] Setting up training...")

# 类别权重（使用focal loss的思想）
class_counts = torch.bincount(data.y[data.train_mask])
class_weights = 1.0 / (class_counts.float() + 1)
# 对小类别给予更高权重
alpha = 0.75
class_weights = class_weights ** alpha
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

criterion = nn.NLLLoss(weight=class_weights)

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.002,
    weight_decay=1e-4
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.005,
    epochs=10,
    steps_per_epoch=1,
    pct_start=0.3
)

# 4. 训练循环
print("\n[4] Starting training...")

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    
    # Top-3准确率
    _, top3_pred = out.topk(3, dim=1)
    top3_correct = 0
    for i in range(mask.sum().item()):
        if data.y[mask][i] in top3_pred[mask][i]:
            top3_correct += 1
    top3_acc = top3_correct / mask.sum().item()
    
    return acc, top3_acc

train_losses = []
val_accs = []
val_top3_accs = []
best_val_acc = 0
patience_counter = 0
patience = 30

epochs = 10
print(f"   Training for {epochs} epochs...")

for epoch in range(1, epochs + 1):
    start_time = time.time()
    
    # 训练
    loss = train()
    train_losses.append(loss)
    
    # 评估
    if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
        train_acc, train_top3 = evaluate(data.train_mask)
        val_acc, val_top3 = evaluate(data.val_mask)
        val_accs.append(val_acc)
        val_top3_accs.append(val_top3)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'experiments/models/best_gat_model.pt')
        else:
            patience_counter += 1
        
        # 打印进度
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} (Top-3: {val_top3:.4f}) | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")
    
    # 早停
    if patience_counter >= patience:
        print(f"   Early stopping at epoch {epoch}")
        break

print(f"   Best validation accuracy: {best_val_acc:.4f}")

# 5. 测试
print("\n[5] Testing model...")
checkpoint = torch.load('experiments/models/best_gat_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_acc, test_top3 = evaluate(data.test_mask)
print(f"   Test accuracy: {test_acc:.4f}")
print(f"   Test top-3 accuracy: {test_top3:.4f}")

# 6. 保存结果
print("\n[6] Saving results...")
os.makedirs('experiments/results', exist_ok=True)

results = {
    'model_type': 'GAT',
    'heads': heads,
    'hidden_channels': hidden_channels,
    'num_parameters': sum(p.numel() for p in model.parameters()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_top3_acc': test_top3,
    'num_epochs': len(train_losses),
    'train_losses': [float(loss) for loss in train_losses],
    'val_accuracies': [float(acc) for acc in val_accs],
    'val_top3_accuracies': [float(acc) for acc in val_top3_accs]
}

with open('experiments/results/gat_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved: experiments/results/gat_results.json")


# 7. 可视化注意力权重（最终修正版）
print("\n[7] Visualizing attention weights...")

model.eval()
with torch.no_grad():
    # 1. 获取注意力权重（处理GATConv的返回值：(output, (edge_index, attn_weights))）
    # 注意：根据模型结构调整conv1的名称（如model.layers[0]、model.gat1等）
    try:
        _, (edge_index, attn_weights) = model.conv1(data.x, data.edge_index, return_attention_weights=True)
    except AttributeError as e:
        # 适配模型层命名不同的情况（示例：若用Sequential包装）
        print(f"Warning: conv1 not found, trying layers[0] | {e}")
        _, (edge_index, attn_weights) = model.layers[0](data.x, data.edge_index, return_attention_weights=True)

    # 2. 处理多头注意力权重：聚合为标量（解决维度问题）
    attn_weights_np = attn_weights.cpu().numpy()  # 形状：[num_edges, num_heads]
    attn_scores = attn_weights_np.mean(axis=1)    # 多头平均，形状：[num_edges,]
    # 可选：过滤极小值（避免数值噪声）
    attn_scores = np.clip(attn_scores, 1e-8, None)  # 替换小于1e-8的值为1e-8

    # 3. 提取边的目标节点（向量化操作）
    dst_nodes = edge_index[1].cpu().numpy()  # 形状：[num_edges,]

    # 4. 计算每个节点的入度注意力分数（向量化累加）
    node_attention = np.zeros(data.num_nodes, dtype=np.float32)
    np.add.at(node_attention, dst_nodes, attn_scores)  # 按目标节点累加

    # 5. 归一化（可选，避免数据范围过小）
    non_zero_mask = node_attention > 1e-8  # 过滤极小值，替代简单的>0
    if np.any(non_zero_mask):
        # 方式：除以全局最大值（将数据缩放到[0,1]，扩大数值范围）
        max_attn = node_attention[non_zero_mask].max()
        node_attention[non_zero_mask] = node_attention[non_zero_mask] / max_attn
        # 可选：对数变换（若数据分布极度偏斜）
        # node_attention[non_zero_mask] = np.log1p(node_attention[non_zero_mask])

# -------------------------- 关键：调试数据分布 --------------------------
print(f"\n--- Attention Weights Statistics ---")
print(f"Non-zero nodes count: {np.sum(non_zero_mask)}")
print(f"Min attention value: {node_attention[non_zero_mask].min():.6f}")
print(f"Max attention value: {node_attention[non_zero_mask].max():.6f}")
print(f"Mean attention value: {node_attention[non_zero_mask].mean():.6f}")
print(f"Unique values count: {len(np.unique(node_attention[non_zero_mask]))}")

# -------------------------- 绘制直方图（鲁棒版） --------------------------
plt.figure(figsize=(10, 6))
data_to_plot = node_attention[non_zero_mask]

# 动态调整bins数量（核心解决bins过多问题）
unique_vals = len(np.unique(data_to_plot))
bins = min(50, unique_vals if unique_vals > 0 else 10)  # 最多50个bins，最少10个
# 若唯一值仍过少，使用固定宽度的bins（如0.01为步长）
if bins < 5:
    bins = np.arange(data_to_plot.min(), data_to_plot.max() + 1e-6, 0.01)

# 绘制直方图
n, bins_used, patches = plt.hist(
    data_to_plot,
    bins=bins,
    alpha=0.7,
    color='purple',
    edgecolor='black'
)

# 添加统计信息标注
plt.text(
    0.95, 0.95,
    f"Non-zero nodes: {np.sum(non_zero_mask)}\nMean: {data_to_plot.mean():.4f}\nMax: {data_to_plot.max():.4f}",
    transform=plt.gca().transAxes,
    ha='right',
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.xlabel('Normalized Attention Weight (Scaled to [0,1])')
plt.ylabel('Number of Nodes')
plt.title('Distribution of GAT Attention Weights (4 Heads, Averaged)')
plt.grid(True, alpha=0.3)

# 确保目录存在并保存
os.makedirs('experiments/visualizations', exist_ok=True)
plt.savefig('experiments/visualizations/gat_attention_dist.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n   Attention weights visualization saved: experiments/visualizations/gat_attention_dist.png")

print("\n" + "=" * 60)
print("GAT TRAINING COMPLETED")
print("=" * 60)