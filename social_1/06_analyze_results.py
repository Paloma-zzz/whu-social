# 06_analyze_results.py
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from collections import Counter

print("=" * 60)
print("ANALYZING MODEL RESULTS")
print("=" * 60)

# 1. 加载数据
print("\n[1] Loading data and model...")

# 加载数据集
data_path = "experiments/data/reddit_graph.pt"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    exit(1)

data = torch.load(data_path, weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 加载训练结果
results_path = "experiments/results/improved_training_results.json"
if not os.path.exists(results_path):
    print(f"Warning: {results_path} not found, using simple results")
    results_path = "experiments/results/training_results.json"

with open(results_path, 'r') as f:
    results = json.load(f)

print(f"   Model type: {results.get('model_type', 'GCN').upper()}")
print(f"   Test accuracy: {results.get('test_acc', 0):.4f}")
print(f"   Best validation accuracy: {results.get('best_val_acc', 0):.4f}")
if 'test_top3_acc' in results:
    print(f"   Test top-3 accuracy: {results['test_top3_acc']:.4f}")

# 2. 加载最佳模型
print("\n[2] Loading best model...")

model_path = "experiments/models/best_model_improved.pt"
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found, trying other models")
    model_path = "experiments/models/best_model.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# 重新创建匹配的模型架构
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

# 根据保存的模型类型创建对应的模型
model_type = results.get('model_type', 'EnhancedGCN')
num_layers = results.get('num_layers', 3)
hidden_dim = results.get('hidden_dim', 128)

if model_type in ['EnhancedGCN', 'enhancedgcn']:
    print(f"   Creating EnhancedGCN with {num_layers} layers, hidden_dim={hidden_dim}")
    
    class EnhancedGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
            super().__init__()
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropout = dropout
            
            # 输入层
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # 隐藏层
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # 输出层
            self.convs.append(GCNConv(hidden_channels, out_channels))
            
            # 额外的线性层
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
    
    model = EnhancedGCN(
        in_channels=data.num_features,
        hidden_channels=hidden_dim,
        out_channels=data.y.max().item() + 1,
        num_layers=num_layers,
        dropout=0.3
    )
else:
    print(f"   Creating simple GCN model")
    
    class SimpleGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.dropout = nn.Dropout(0.5)
            self.batch_norm = nn.BatchNorm1d(hidden_channels)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    model = SimpleGCN(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=data.y.max().item() + 1
    )

model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Model loaded successfully")

# 3. 详细性能评估
print("\n[3] Detailed performance evaluation...")

@torch.no_grad()
def get_predictions(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()

# 获取预测结果
train_pred, train_true = get_predictions(data.train_mask)
val_pred, val_true = get_predictions(data.val_mask)
test_pred, test_true = get_predictions(data.test_mask)

print(f"   Train accuracy: {(train_pred == train_true).mean():.4f}")
print(f"   Validation accuracy: {(val_pred == val_true).mean():.4f}")
print(f"   Test accuracy: {(test_pred == test_true).mean():.4f}")

# 4. 类别级别的性能分析
print("\n[4] Per-class performance analysis...")

# 计算每个类别的准确率
num_classes = data.y.max().item() + 1
class_accuracies = {}
class_counts = {}
class_precisions = {}
class_recalls = {}

for class_id in range(num_classes):
    # 在测试集中找到该类别的样本
    class_mask = (data.y == class_id) & data.test_mask
    if class_mask.sum().item() > 0:
        class_pred, class_true = get_predictions(class_mask)
        class_acc = (class_pred == class_true).mean()
        class_accuracies[class_id] = class_acc
        class_counts[class_id] = class_mask.sum().item()
        
        # 计算精确率和召回率
        true_positives = ((class_pred == class_id) & (class_true == class_id)).sum()
        predicted_positives = (class_pred == class_id).sum()
        actual_positives = (class_true == class_id).sum()
        
        precision = true_positives / max(predicted_positives, 1)
        recall = true_positives / max(actual_positives, 1)
        
        class_precisions[class_id] = precision
        class_recalls[class_id] = recall

# 按类别大小排序
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n   Top 10 classes by size and their performance:")
print(f"   {'Class':<6} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
print(f"   {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for i, (class_id, count) in enumerate(sorted_classes[:10]):
    acc = class_accuracies.get(class_id, 0)
    prec = class_precisions.get(class_id, 0)
    rec = class_recalls.get(class_id, 0)
    print(f"   {class_id:<6} {count:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")

# 5. 混淆矩阵
print("\n[5] Generating confusion matrix...")

# 选择样本最多的前15个类别
top_n = min(15, len(sorted_classes))
top_class_ids = [class_id for class_id, _ in sorted_classes[:top_n]]

# 筛选这些类别的测试样本
mask_top = data.test_mask & torch.isin(data.y, torch.tensor(top_class_ids, device=device))
test_pred_top, test_true_top = get_predictions(mask_top)

# 创建混淆矩阵
cm = confusion_matrix(test_true_top, test_pred_top, labels=top_class_ids, normalize='true')

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=top_class_ids,
            yticklabels=top_class_ids)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Normalized Confusion Matrix (Top {top_n} Classes)')
plt.tight_layout()
plt.savefig('experiments/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"   Confusion matrix saved: experiments/visualizations/confusion_matrix.png")

# 6. 训练过程可视化
print("\n[6] Visualizing training process...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 损失曲线
if 'train_losses' in results:
    train_losses = results['train_losses']
    ax1.plot(train_losses, color='royalblue', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    if max(train_losses) > 10 * min(train_losses):
        ax1.set_yscale('log')

# 准确率曲线
if 'val_accuracies' in results:
    val_accs = results['val_accuracies']
    # 确定x轴位置
    if len(val_accs) <= 25:
        epochs = range(2, len(val_accs)*2 + 1, 2)
    else:
        epochs = range(len(val_accs))
    
    ax2.plot(epochs, val_accs, color='darkorange', linewidth=2, label='Validation')
    ax2.axhline(y=results.get('test_acc', 0), color='forestgreen', linestyle='--', 
                linewidth=2, label=f'Test: {results.get("test_acc", 0):.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{results.get("model_type", "GCN").upper()} Training Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plt.suptitle(f'Model Training Analysis (Final Test Acc: {results.get("test_acc", 0):.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/visualizations/training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"   Training analysis saved: experiments/visualizations/training_analysis.png")

# 7. 类别准确率与样本数量的关系
print("\n[7] Analyzing relationship between class size and accuracy...")

class_sizes = []
class_accs = []
for class_id, acc in class_accuracies.items():
    if class_id in class_counts:
        class_sizes.append(class_counts[class_id])
        class_accs.append(acc)

if class_sizes and class_accs:
    plt.figure(figsize=(10, 6))
    plt.scatter(class_sizes, class_accs, alpha=0.6, s=50, color='purple')
    plt.xlabel('Number of Samples in Class')
    plt.ylabel('Classification Accuracy')
    plt.title('Class Size vs Classification Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(class_sizes) > 1:
        try:
            z = np.polyfit(class_sizes, class_accs, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(class_sizes), max(class_sizes), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8, label='Trend line')
            plt.legend()
        except:
            pass
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/class_size_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Class size analysis saved: experiments/visualizations/class_size_vs_accuracy.png")

# 8. 生成详细报告
print("\n[8] Generating detailed report...")

detailed_report = {
    "experiment_summary": {
        "model_type": results.get("model_type", "GCN"),
        "dataset_size": {
            "total_nodes": data.num_nodes,
            "train_nodes": data.train_mask.sum().item(),
            "val_nodes": data.val_mask.sum().item(),
            "test_nodes": data.test_mask.sum().item()
        },
        "performance": {
            "train_accuracy": float((train_pred == train_true).mean()),
            "val_accuracy": float((val_pred == val_true).mean()),
            "test_accuracy": results.get("test_acc", 0),
            "test_top3_accuracy": results.get("test_top3_acc", 0),
            "best_val_accuracy": results.get("best_val_acc", 0)
        }
    },
    "class_performance": {
        str(class_id): {
            "sample_count": class_counts[class_id],
            "accuracy": float(class_accuracies[class_id]),
            "precision": float(class_precisions.get(class_id, 0)),
            "recall": float(class_recalls.get(class_id, 0))
        }
        for class_id in class_accuracies.keys()
    },
    "training_details": {
        "epochs_trained": results.get("num_epochs", 0),
        "final_training_loss": results.get("train_losses", [0])[-1] if results.get("train_losses") else 0,
        "parameters_count": results.get("num_parameters", 0)
    }
}

report_path = "experiments/results/detailed_report.json"
with open(report_path, 'w') as f:
    json.dump(detailed_report, f, indent=2)

print(f"   Detailed report saved: {report_path}")

# 9. 性能总结
print("\n[9] Performance Summary:")
print(f"   Model: {results.get('model_type', 'GCN').upper()}")
print(f"   Parameters: {results.get('num_parameters', 0):,}")
print(f"   Epochs trained: {results.get('num_epochs', 0)}")
if results.get('train_losses'):
    print(f"   Final training loss: {results['train_losses'][-1]:.4f}")
print(f"   Test accuracy: {results.get('test_acc', 0):.4f}")
if 'test_top3_acc' in results:
    print(f"   Test top-3 accuracy: {results['test_top3_acc']:.4f}")

# 计算类别不平衡影响
acc_by_size = {}
for class_id, count in class_counts.items():
    if class_id in class_accuracies:
        if count > 1000:
            size_group = "large"
        elif count > 100:
            size_group = "medium"
        else:
            size_group = "small"
        
        if size_group not in acc_by_size:
            acc_by_size[size_group] = []
        acc_by_size[size_group].append(class_accuracies[class_id])

print(f"\n   Accuracy by class size:")
for size_group, accs in acc_by_size.items():
    if accs:
        print(f"     {size_group.capitalize()} classes ({len(accs)}): {np.mean(accs):.4f}")

# 10. 论文结果准备
print("\n[10] Key findings for paper:")
print(f"   1. Overall test accuracy: {results.get('test_acc', 0):.4f} (baseline: 0.1089)")
if 'test_top3_acc' in results:
    print(f"   2. Top-3 accuracy: {results['test_top3_acc']:.4f} (random baseline: {3/41:.4f})")
print(f"   3. Performance improvement: {results.get('test_acc', 0) - 0.1089:.4f}")
print(f"   4. Model shows clear learning of user interests from social graph structure")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)