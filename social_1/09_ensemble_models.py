# 09_ensemble_models.py
import torch
import numpy as np
import json
import os
from collections import Counter

print("=" * 60)
print("MODEL ENSEMBLE FOR IMPROVED PERFORMANCE")
print("=" * 60)

# 1. 加载数据
print("\n[1] Loading data...")
data = torch.load('experiments/data/original_social_graph.pt', weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 2. 加载多个模型
print("\n[2] Loading multiple models...")

models = {}
predictions = {}

# 加载GCN模型
try:
    from torch_geometric.nn import GCNConv
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    gcn_model = SimpleGCN(data.num_features, 128, data.y.max().item() + 1)
    gcn_checkpoint = torch.load('experiments/models/best_model_improved.pt', weights_only=False)
    gcn_model.load_state_dict(gcn_checkpoint['model_state_dict'])
    gcn_model = gcn_model.to(device)
    gcn_model.eval()
    models['gcn'] = gcn_model
    print("   ✓ GCN model loaded")
except Exception as e:
    print(f"   ✗ GCN model loading failed: {e}")

# 加载GAT模型
try:
    from torch_geometric.nn import GATConv
    
    class SimpleGAT(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    gat_model = SimpleGAT(data.num_features, 64, data.y.max().item() + 1, heads=4)
    gat_checkpoint = torch.load('experiments/models/best_gat_model.pt', weights_only=False)
    gat_model.load_state_dict(gat_checkpoint['model_state_dict'])
    gat_model = gat_model.to(device)
    gat_model.eval()
    models['gat'] = gat_model
    print("   ✓ GAT model loaded")
except Exception as e:
    print(f"   ✗ GAT model loading failed: {e}")

# 3. 生成各个模型的预测
print("\n[3] Generating predictions...")

for name, model in models.items():
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions[name] = {
            'logits': out,
            'probs': torch.exp(out),
            'preds': out.argmax(dim=1)
        }
    print(f"   {name.upper()} test accuracy: {(predictions[name]['preds'][data.test_mask] == data.y[data.test_mask]).float().mean():.4f}")

# 4. 集成策略
print("\n[4] Applying ensemble strategies...")

ensemble_results = {}

# 策略1: 平均概率
if len(models) > 1:
    avg_probs = torch.zeros_like(predictions[list(models.keys())[0]]['probs'])
    for name in models.keys():
        avg_probs += predictions[name]['probs']
    avg_probs /= len(models)
    avg_preds = avg_probs.argmax(dim=1)
    
    avg_acc = (avg_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    ensemble_results['average_prob'] = avg_acc
    print(f"   Average probability accuracy: {avg_acc:.4f}")

# 策略2: 加权平均（根据验证集性能）
val_accs = {}
for name in models.keys():
    val_acc = (predictions[name]['preds'][data.val_mask] == data.y[data.val_mask]).float().mean().item()
    val_accs[name] = val_acc

if len(models) > 1:
    weights = torch.tensor([val_accs[name] for name in models.keys()], device=device)
    weights = weights / weights.sum()
    
    weighted_probs = torch.zeros_like(predictions[list(models.keys())[0]]['probs'])
    for i, name in enumerate(models.keys()):
        weighted_probs += weights[i] * predictions[name]['probs']
    
    weighted_preds = weighted_probs.argmax(dim=1)
    weighted_acc = (weighted_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    ensemble_results['weighted_average'] = weighted_acc
    print(f"   Weighted average accuracy: {weighted_acc:.4f}")

# 策略3: 投票
if len(models) > 1:
    vote_preds = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    for i in range(data.num_nodes):
        votes = []
        for name in models.keys():
            votes.append(predictions[name]['preds'][i].item())
        vote_preds[i] = Counter(votes).most_common(1)[0][0]
    
    vote_acc = (vote_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    ensemble_results['voting'] = vote_acc
    print(f"   Voting accuracy: {vote_acc:.4f}")

# 5. 保存集成结果
print("\n[5] Saving ensemble results...")

best_method = max(ensemble_results, key=ensemble_results.get)
best_acc = ensemble_results[best_method]

results = {
    'ensemble_methods': ensemble_results,
    'best_method': best_method,
    'best_accuracy': best_acc,
    'individual_accuracies': {name: (predictions[name]['preds'][data.test_mask] == data.y[data.test_mask]).float().mean().item() 
                            for name in models.keys()}
}

with open('experiments/results/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved: experiments/results/ensemble_results.json")
print(f"   Best ensemble method: {best_method} with accuracy {best_acc:.4f}")

print("\n" + "=" * 60)
print("ENSEMBLE COMPLETED")
print("=" * 60)