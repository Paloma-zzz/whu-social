import torch
import torch.nn as nn
from transformers import BertModel
from torch_geometric.nn import GCNConv

class TextEncoder(nn.Module):
    """文本编码器：BERT"""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["model"]["bert_model_name"])
        self.dropout = nn.Dropout(config["model"]["dropout"])
        self.fc = nn.Linear(config["model"]["bert_hidden_dim"], config["model"]["bert_hidden_dim"])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.pooler_output  # [batch_size, bert_hidden_dim]
        cls_emb = self.dropout(cls_emb)
        return self.fc(cls_emb)

class BehaviorEncoder(nn.Module):
    """行为编码器：双向LSTM"""
    def __init__(self, config):
        super().__init__()
        self.seq_len = config["data"]["behavior_seq_len"]
        self.hidden_dim = config["model"]["lstm_hidden_dim"]
        # LSTM：input_size=1（每个时间步的特征维度是1）
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config["model"]["dropout"]
        )
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # 双向输出拼接
        self.dropout = nn.Dropout(config["model"]["dropout"])

    def forward(self, x):
        # x: [batch_size, seq_len] → [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x)
        # 取最后一层的双向隐藏状态
        hn = torch.cat([hn[-2], hn[-1]], dim=-1)  # [batch_size, hidden_dim*2]
        out = self.fc(hn)
        return self.dropout(out)

class GraphEncoder(nn.Module):
    """图编码器：GCN"""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["data"]["behavior_seq_len"]  # 节点特征维度=行为序列长度
        self.hidden_dim = config["model"]["gcn_hidden_dim"]
        self.output_dim = config["model"]["gcn_hidden_dim"]
        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gcn2 = GCNConv(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(config["model"]["dropout"])
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        return self.gcn2(x, edge_index)

class AttentionFusion(nn.Module):
    """注意力机制：多模态特征融合"""
    def __init__(self, config):
        super().__init__()
        self.text_dim = config["model"]["bert_hidden_dim"]
        self.behavior_dim = config["model"]["lstm_hidden_dim"]
        self.graph_dim = config["model"]["gcn_hidden_dim"]
        self.fusion_dim = config["model"]["fusion_hidden_dim"]

        # 注意力权重计算层
        self.attention = nn.Linear(self.text_dim + self.behavior_dim + self.graph_dim, 3)
        # 融合后全连接层
        self.fc = nn.Linear(self.text_dim + self.behavior_dim + self.graph_dim, self.fusion_dim)
        self.dropout = nn.Dropout(config["model"]["dropout"])
        self.relu = nn.ReLU()

    def forward(self, text_feat, behavior_feat, graph_feat):
        # 拼接特征：[batch_size, sum_dim]
        concat_feat = torch.cat([text_feat, behavior_feat, graph_feat], dim=-1)
        # 计算注意力权重：[batch_size, 3]
        attn_weights = torch.softmax(self.attention(concat_feat), dim=-1)
        # 加权融合各模态特征
        text_weighted = text_feat * attn_weights[:, 0:1]
        behavior_weighted = behavior_feat * attn_weights[:, 1:2]
        graph_weighted = graph_feat * attn_weights[:, 2:3]
        fused_feat = torch.cat([text_weighted, behavior_weighted, graph_weighted], dim=-1)
        # 全连接层优化
        fused_feat = self.relu(self.fc(fused_feat))
        return self.dropout(fused_feat), attn_weights

class MultiModalUserProfile(nn.Module):
    """多模态用户画像主模型"""
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.behavior_encoder = BehaviorEncoder(config)
        self.graph_encoder = GraphEncoder(config)
        self.fusion = AttentionFusion(config)
        # 分类头（辅助训练，预测用户标签）
        self.classifier = nn.Linear(config["model"]["fusion_hidden_dim"], config["data"]["n_classes"])

    def forward(self, text_input, behavior_input, graph_data, user_idx):
        """
        参数说明：
        - text_input: dict → input_ids, attention_mask
        - behavior_input: [batch_size, seq_len]
        - graph_data: PyG Data → x, edge_index
        - user_idx: [batch_size] → 用户索引（提取图特征）
        """
        # 1. 文本特征提取
        text_feat = self.text_encoder(text_input["input_ids"], text_input["attention_mask"])
        # 2. 行为特征提取
        behavior_feat = self.behavior_encoder(behavior_input)
        # 3. 图特征提取（全局→批量）
        graph_feat_global = self.graph_encoder(graph_data.x, graph_data.edge_index)
        graph_feat = graph_feat_global[user_idx]
        # 4. 特征融合
        fused_feat, attn_weights = self.fusion(text_feat, behavior_feat, graph_feat)
        # 5. 标签预测
        logits = self.classifier(fused_feat)
        return logits, fused_feat, attn_weights