# 05_train_gnn.py (修正版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

print("=" * 60)
print("TRAINING GNN MODEL")
print("=" * 60)

# 1. 加载数据
print("\n[1] Loading dataset...")
data_path = "experiments/data/reddit_graph.pt"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    print("Please run 04_build_graph.py first")
    exit(1)

# 修正：使用 weights_only=False 加载数据
try:
    data = torch.load(data_path, weights_only=False)
    print("   Dataset loaded with weights_only=False")
except Exception as e:
    print(f"   Error loading with weights_only=False: {e}")
    print("   Trying alternative loading method...")
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

print(f"   Dataset loaded:")
print(f"     Nodes: {data.num_nodes:,}")
print(f"     Edges: {data.num_edges:,}")
print(f"     Features: {data.num_features}")
print(f"     Classes: {data.y.max().item() + 1}")
print(f"     Train nodes: {data.train_mask.sum().item():,}")
print(f"     Val nodes:   {data.val_mask.sum().item():,}")
print(f"     Test nodes:  {data.test_mask.sum().item():,}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

data = data.to(device)

# 2. 定义GNN模型
print("\n[2] Building GNN model...")

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    print("   ✓ GNN modules imported successfully")
except ImportError as e:
    print(f"   ✗ Error importing GNN modules: {e}")
    print("   Please install torch_geometric: pip install torch_geometric")
    exit(1)

class GNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.batch_norm = nn.BatchNorm1d(hidden_channels)
        elif model_type == 'gat':
            self.heads = 4  # GAT多头注意力
            self.conv1 = GATConv(in_channels, hidden_channels, heads=self.heads, dropout=0.6)
            self.conv2 = GATConv(hidden_channels * self.heads, out_channels, heads=1, concat=False, dropout=0.6)
            self.batch_norm = nn.BatchNorm1d(hidden_channels * self.heads)  # 修正：维度为 hidden_channels * heads
        elif model_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            self.batch_norm = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

# 3. 选择模型类型
print("\n[3] Select model architecture:")
print("   1. GCN (Graph Convolutional Network)")
print("   2. GAT (Graph Attention Network)")
print("   3. GraphSAGE")

model_choice = input("   Enter choice (1/2/3, default=1): ").strip() or "1"

model_types = {'1': 'gcn', '2': 'gat', '3': 'sage'}
model_type = model_types.get(model_choice, 'gcn')

in_channels = data.num_features
hidden_channels = 128
out_channels = data.y.max().item() + 1

model = GNNClassifier(in_channels, hidden_channels, out_channels, model_type=model_type)
model = model.to(device)

print(f"   Model: {model_type.upper()}")
if model_type == 'gat':
    print(f"   GAT heads: {model.heads}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 4. 训练设置
print("\n[4] Training setup...")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 类别权重（处理不平衡）
class_counts = torch.bincount(data.y[data.train_mask])
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

criterion = nn.NLLLoss(weight=class_weights)

# 5. 训练循环
print("\n[5] Starting training...")

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc

train_losses = []
val_accs = []
best_val_acc = 0
patience_counter = 0
patience = 20

# 根据模型类型调整epoch数
if model_type == 'gat':
    epochs = 30  # GAT训练较慢，减少epoch数
else:
    epochs = 50

print(f"   Training for {epochs} epochs...")

for epoch in range(1, epochs + 1):
    start_time = time.time()
    
    # 训练
    loss = train()
    train_losses.append(loss)
    
    # 评估（每5个epoch评估一次以减少计算）
    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        train_acc = evaluate(data.train_mask)
        val_acc = evaluate(data.val_mask)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step(loss)
        
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
                'model_type': model_type,
                'hidden_channels': hidden_channels
            }, 'experiments/models/best_model.pt')
        else:
            patience_counter += 1
        
        # 打印进度
        elapsed = time.time() - start_time
        print(f"   Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")
    
    # 早停
    if patience_counter >= patience:
        print(f"   Early stopping at epoch {epoch}")
        break

print(f"   Best validation accuracy: {best_val_acc:.4f}")

# 6. 测试模型
print("\n[6] Testing model...")
try:
    checkpoint = torch.load('experiments/models/best_model.pt', weights_only=False)
    # 检查模型类型是否匹配
    if 'model_type' in checkpoint and checkpoint['model_type'] == model_type:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("   Warning: Model type mismatch, using current model")
    test_acc = evaluate(data.test_mask)
    print(f"   Test accuracy: {test_acc:.4f}")
except Exception as e:
    print(f"   Error loading best model: {e}")
    print("   Using current model for testing...")
    test_acc = evaluate(data.test_mask)
    print(f"   Test accuracy: {test_acc:.4f}")

# 7. 保存结果
print("\n[7] Saving results...")

os.makedirs('experiments/results', exist_ok=True)

results = {
    'model_type': model_type,
    'num_parameters': sum(p.numel() for p in model.parameters()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'num_epochs': len(train_losses),
    'train_losses': [float(loss) for loss in train_losses],
    'val_accuracies': [float(acc) for acc in val_accs]
}

with open('experiments/results/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved: experiments/results/training_results.json")

# 8. 可视化训练过程
print("\n[8] Creating training visualization...")

os.makedirs('experiments/visualizations', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 损失曲线
ax1.plot(train_losses, color='royalblue', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Curve')
ax1.grid(True, alpha=0.3)
if max(train_losses) > 10 * min(train_losses):
    ax1.set_yscale('log')

# 准确率曲线
# 注意：val_accs只在某些epoch记录
if val_accs:
    val_epochs = range(1, len(val_accs)*5, 5)
    if len(val_epochs) > len(val_accs):
        val_epochs = val_epochs[:len(val_accs)]
    ax2.plot(val_epochs, val_accs, color='darkorange', 
             linewidth=2, marker='o', markersize=4, label='Validation')
ax2.axhline(y=test_acc, color='forestgreen', linestyle='--', 
            linewidth=2, label=f'Test: {test_acc:.4f}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title(f'{model_type.upper()} Performance')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle(f'GNN Training Results (Best Val Acc: {best_val_acc:.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/visualizations/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   Visualization saved: experiments/visualizations/training_curves.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)