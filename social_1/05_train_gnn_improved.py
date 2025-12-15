# 05_train_gnn_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt

print("=" * 60)
print("IMPROVED GNN TRAINING")
print("=" * 60)

# 1. 加载数据并预处理
print("\n[1] Loading and preprocessing dataset...")

data_path = "experiments/data/reddit_graph.pt"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    exit(1)

# 方法1: 尝试使用torch.load并处理版本兼容性
try:
    # PyTorch 2.6+ 需要处理weights_only
    import torch.serialization
    # 添加安全全局变量
    torch.serialization.add_safe_globals([torch._C._TensorMeta])
    
    # 尝试加载
    data = torch.load(data_path, weights_only=False)
    print("   Dataset loaded with torch.load(weights_only=False)")
    
except Exception as e:
    print(f"   Error with torch.load: {e}")
    print("   Trying alternative loading method...")
    
    # 方法2: 使用pickle直接加载
    import pickle
    try:
        with open(data_path, 'rb') as f:
            # 尝试不同的pickle协议
            data = pickle.load(f)
        print("   Dataset loaded with pickle.load()")
    except Exception as e2:
        print(f"   Error with pickle: {e2}")
        print("   Trying one more method...")
        
        # 方法3: 重新构建数据
        print("   Reconstructing dataset from components...")
        
        # 加载保存的统计信息和特征
        with open('experiments/data/dataset_stats.json', 'r') as f:
            stats = json.load(f)
        
        # 重新加载特征矩阵
        features = np.load('processed_data/features.npy')
        
        # 创建简化版的数据集（部分节点）
        num_nodes = min(50000, stats['num_nodes'])  # 使用5万节点
        indices = np.random.choice(features.shape[0], num_nodes, replace=False)
        features = features[indices]
        
        # 加载类别映射
        with open('processed_data/class_map_processed.json', 'r') as f:
            class_map = json.load(f)
        
        with open('processed_data/id_map_processed.json', 'r') as f:
            id_map = json.load(f)
        
        # 创建标签数组
        labels = np.zeros(num_nodes, dtype=np.int64)
        for i, idx in enumerate(indices):
            node_id = list(id_map.keys())[list(id_map.values()).index(idx)]
            if node_id in class_map:
                labels[i] = class_map[node_id]
        
        # 创建简单的图结构（KNN图）
        from sklearn.neighbors import kneighbors_graph
        print("   Building KNN graph...")
        adj_matrix = kneighbors_graph(features, n_neighbors=20, mode='connectivity', include_self=False)
        edges = adj_matrix.nonzero()
        edge_index = np.vstack([edges[0], edges[1]])
        
        # 创建PyG数据对象
        from torch_geometric.data import Data
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long)
        )
        
        # 划分数据集
        from sklearn.model_selection import train_test_split
        node_indices = list(range(num_nodes))
        train_idx, temp_idx = train_test_split(
            node_indices, test_size=0.3, stratify=labels, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
        )
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        print("   Dataset reconstructed successfully")

print(f"   Dataset loaded:")
print(f"     Nodes: {data.num_nodes:,}")
print(f"     Edges: {data.num_edges:,}")
print(f"     Features: {data.num_features}")
print(f"     Classes: {data.y.max().item() + 1}")

# 特征标准化
print("   Standardizing features...")
mean = data.x.mean(dim=0, keepdim=True)
std = data.x.std(dim=0, keepdim=True) + 1e-8  # 避免除零
data.x = (data.x - mean) / std
print(f"   Feature range after standardization: [{data.x.min():.3f}, {data.x.max():.3f}]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")
data = data.to(device)

# 2. 增强的GCN模型
print("\n[2] Building enhanced GCN model...")

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import BatchNorm
    print("   ✓ GCN modules imported successfully")
except ImportError:
    print("   ✗ torch_geometric not installed properly")
    print("   Please install: pip install torch_geometric")
    exit(1)

class EnhancedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout
        
        # 输入层
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))  # 使用标准BatchNorm
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 输出层
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # 额外的线性层用于提高表达能力
        self.fc = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index):
        # 多层GCN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        # 额外的全连接层
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

# 模型参数
in_channels = data.num_features
hidden_channels = 128  # 减少隐藏层维度以加快训练
out_channels = data.y.max().item() + 1
num_layers = 3  # 减少层数

model = EnhancedGCN(in_channels, hidden_channels, out_channels, 
                    num_layers=num_layers, dropout=0.3)  # 减少dropout
model = model.to(device)

print(f"   Model: Enhanced GCN with {num_layers} layers")
print(f"   Hidden dimension: {hidden_channels}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. 改进的训练设置
print("\n[3] Setting up training...")

# 类别权重（处理不平衡）
class_counts = torch.bincount(data.y[data.train_mask])
# 使用逆频率加权，但对小类别进行截断避免过大权重
class_weights = 1.0 / torch.sqrt(class_counts.float() + 1)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

criterion = nn.NLLLoss(weight=class_weights)

# 优化器：AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001,  # 更小的学习率
    weight_decay=5e-5  # 权重衰减
)

# 学习率调度器：余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50,  # 总epoch数
    eta_min=1e-5  # 最小学习率
)

# 4. 训练循环
print("\n[4] Starting training...")

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 添加L2正则化
    l2_reg = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss = loss + 1e-4 * l2_reg
    
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    
    # 计算top-3准确率
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
patience = 20

epochs = 50
print(f"   Training for {epochs} epochs...")

for epoch in range(1, epochs + 1):
    start_time = time.time()
    
    # 训练
    loss = train()
    train_losses.append(loss)
    
    # 评估（每2个epoch评估一次）
    if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
        train_acc, train_top3 = evaluate(data.train_mask)
        val_acc, val_top3 = evaluate(data.val_mask)
        val_accs.append(val_acc)
        val_top3_accs.append(val_top3)
        
        # 学习率调整
        scheduler.step()
        
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
            }, 'experiments/models/best_model_improved.pt')
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

# 5. 测试模型
print("\n[5] Testing model...")
try:
    checkpoint = torch.load('experiments/models/best_model_improved.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc, test_top3 = evaluate(data.test_mask)
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test top-3 accuracy: {test_top3:.4f}")
except Exception as e:
    print(f"   Error loading best model: {e}")
    print("   Using current model for testing...")
    test_acc, test_top3 = evaluate(data.test_mask)
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test top-3 accuracy: {test_top3:.4f}")

# 6. 保存结果
print("\n[6] Saving results...")
os.makedirs('experiments/results', exist_ok=True)

results = {
    'model_type': 'EnhancedGCN',
    'num_layers': num_layers,
    'hidden_dim': hidden_channels,
    'num_parameters': sum(p.numel() for p in model.parameters()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_top3_acc': test_top3,
    'num_epochs': len(train_losses),
    'train_losses': [float(loss) for loss in train_losses],
    'val_accuracies': [float(acc) for acc in val_accs],
    'val_top3_accuracies': [float(acc) for acc in val_top3_accs]
}

with open('experiments/results/improved_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved: experiments/results/improved_training_results.json")

# 7. 可视化
print("\n[7] Creating visualizations...")
os.makedirs('experiments/visualizations', exist_ok=True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 损失曲线
ax1.plot(train_losses, color='royalblue', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Curve')
ax1.grid(True, alpha=0.3)
if max(train_losses) > 10 * min(train_losses):
    ax1.set_yscale('log')

# 准确率曲线
epochs_to_plot = range(2, len(val_accs)*2 + 1, 2)
ax2.plot(epochs_to_plot, val_accs, color='darkorange', linewidth=2, label='Validation')
ax2.plot(epochs_to_plot, val_top3_accs, color='green', linewidth=2, linestyle='--', label='Validation Top-3')
ax2.axhline(y=test_acc, color='forestgreen', linestyle='--', linewidth=2, label=f'Test: {test_acc:.4f}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Performance')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 类别分布（简化）
with open('experiments/data/dataset_stats.json', 'r') as f:
    stats = json.load(f)

class_counts = [stats['class_distribution'][str(i)] for i in range(min(20, stats['num_classes']))]
ax3.bar(range(len(class_counts)), class_counts, alpha=0.7, color='steelblue')
ax3.set_xlabel('Class ID')
ax3.set_ylabel('Number of Nodes')
ax3.set_title('Class Distribution (Top 20)')
ax3.grid(True, alpha=0.3)

# 预测置信度分布
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = torch.exp(out[data.test_mask])
    max_probs = probs.max(dim=1)[0].cpu().numpy()

ax4.hist(max_probs, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax4.set_xlabel('Prediction Confidence')
ax4.set_ylabel('Frequency')
ax4.set_title('Prediction Confidence Distribution')
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Enhanced GCN Training Results (Test Acc: {test_acc:.4f}, Top-3: {test_top3:.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/visualizations/improved_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Visualization saved: experiments/visualizations/improved_training_results.png")

print("\n" + "=" * 60)
print("IMPROVED TRAINING COMPLETED")
print("=" * 60)

# 8. 性能分析
print("\n[8] Performance Analysis:")
print(f"   Model improvement: {test_acc - 0.1089:.4f} (from {0.1089:.4f} to {test_acc:.4f})")
print(f"   Top-3 accuracy: {test_top3:.4f} (41 classes random: {3/41:.4f})")

if test_top3 > 0.3:
    print("   ✓ Model shows reasonable performance in top-3 prediction")
else:
    print("   ✗ Model performance still needs improvement")

# 建议下一步
print("\n[9] Next Steps:")
print("   1. If accuracy > 30%: Run 06_analyze_results.py for detailed analysis")
print("   2. If accuracy < 30%: Consider using original graph structure")
print("   3. Try different model architectures (GAT, GraphSAGE)")
print("   4. Implement more advanced techniques (graph attention, residual connections)")