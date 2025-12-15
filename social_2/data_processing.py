import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from torch.utils.data import Dataset, DataLoader
import os  # 新增：导入os模块，用于文件检查
import numpy as np
import json  # 新增：用于保存字典（邻接表）

class SocialNetworkDataset:
    def __init__(self, config):
        self.config = config
        self.user_num = config["data"]["user_num"]  # 对应步骤1的num_users（如1000）
        self.n_classes = config["data"]["n_classes"]  # 用户标签数（如2：积极/消极用户）
        self.text_max_len = config["data"]["text_max_len"]
        self.behavior_seq_len = config["data"]["behavior_seq_len"]  # 原代码的16
        
        # 1. 读取带用户ID的微博数据集
        self.df = pd.read_csv("weibo_senti_100k_with_userid.csv")
        # 按user_id分组，得到每个用户的所有微博
        self.user_groups = self.df.groupby("user_id")
        self.user_ids = list(self.user_groups.groups.keys())
        self.user_ids = self.user_ids[:self.user_num]  # 只取前50个用户
        self.user_groups = self.df[self.df["user_id"].isin(self.user_ids)].groupby("user_id")
        
        # 初始化BERT分词器（注意：换成DistilBERT的分词器）
        from transformers import BertTokenizer  # 仅保留BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_model_name"])
        
        # 3. 预处理各模态数据
        self.texts = self._preprocess_user_text()  # 每个用户的文本（所有微博拼接）
        self.behaviors = self._preprocess_user_behavior()  # 每个用户的行为序列
        self.labels = self._preprocess_user_label()  # 每个用户的标签（积极/消极）
        self.adjacency = self._build_social_relation()  # 模拟社交关系（基于文本相似度）
        
        # 4. 预处理图数据（适配原代码的GCN）
        self.graph_data = self._preprocess_graph()
        
        # 5. 划分训练/测试集
        self.train_idx, self.test_idx = self._split_data()

    def _preprocess_user_text(self):
        """每个用户的文本：拼接该用户的所有微博"""
        texts = []
        for user_id in self.user_ids:
            user_weibos = self.user_groups.get_group(user_id)["review"].tolist()
            user_text = " ".join(user_weibos)  # 所有微博拼接成一个长文本
            texts.append(user_text)
        return texts

    def _preprocess_user_behavior(self):
        """每个用户的行为序列：用微博的情感标签（1→点赞，0→评论），取最近16条"""
        behaviors = []
        for user_id in self.user_ids:
            # 取该用户的所有情感标签
            user_labels = self.user_groups.get_group(user_id)["label"].tolist()
            # 转换为行为编码：1=点赞，0=评论
            behavior_seq = [1 if label == 1 else 0 for label in user_labels]
            # 截取/补齐到behavior_seq_len（如16）
            if len(behavior_seq) > self.behavior_seq_len:
                behavior_seq = behavior_seq[-self.behavior_seq_len:]  # 取最近的16条
            else:
                behavior_seq += [0] * (self.behavior_seq_len - len(behavior_seq))  # 不足补0
            behaviors.append(behavior_seq)
        # 标准化行为数据
        scaler = StandardScaler()
        return scaler.fit_transform(behaviors)

    def _preprocess_user_label(self):
        """每个用户的标签：正向微博占比>60%→1（积极用户），否则→0（消极用户）"""
        labels = []
        for user_id in self.user_ids:
            user_labels = self.user_groups.get_group(user_id)["label"].tolist()
            positive_ratio = sum(user_labels) / len(user_labels)
            labels.append(1 if positive_ratio > 0.6 else 0)
        return labels

    

    def _build_social_relation(self):
        """模拟社交关系：基于用户文本的相似度，添加特征复用逻辑"""
        # 定义特征保存路径
        feature_save_path = "user_features.npz"
        adjacency_save_path = "user_adjacency.json"

        # 第一步：检查是否有保存的特征文件，有则直接加载
        if os.path.exists(feature_save_path) and os.path.exists(adjacency_save_path):
            print(f"加载已保存的用户特征和社交关系...")
            # 加载BERT特征
            data = np.load(feature_save_path)
            user_features = data["user_features"]
            self.user_ids = data["user_ids"].tolist()
            # 加载邻接表（社交关系）
            with open(adjacency_save_path, "r", encoding="utf-8") as f:
                adjacency = json.load(f)
                # 将json中的字符串key转换为int（因为user_id是int）
                adjacency = {int(k): v for k, v in adjacency.items()}
            return adjacency

        # 第二步：没有保存文件，执行BERT特征提取（原逻辑）
        from transformers import BertModel
        from tqdm import tqdm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = BertModel.from_pretrained(self.config["model"]["bert_model_name"]).to(device)
        bert_model.eval()

        user_features = []
        for text in tqdm(self.texts, desc="提取用户文本特征（BERT）"):
            encoded = self.tokenizer(
                text,
                max_length=self.text_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = bert_model(**encoded)
            user_features.append(output.pooler_output.cpu().numpy().squeeze())

        # 转换为numpy数组
        user_features = np.array(user_features)
        user_ids_np = np.array(self.user_ids)

        # 第三步：计算相似度并构建邻接表（原逻辑）
        sim_matrix = cosine_similarity(user_features)
        adjacency = defaultdict(list)
        for i, user_id in enumerate(self.user_ids):
            sim_scores = sim_matrix[i]
            sim_scores[i] = -1  # 排除自己
            top_neighbor_idx = sim_scores.argsort()[-random.randint(2, 5):][::-1]
            adjacency[user_id] = [self.user_ids[idx] for idx in top_neighbor_idx]

        # 第四步：保存特征和邻接表到文件（复用的核心）
        print(f"保存用户特征和社交关系到文件...")
        # 保存numpy数组（特征、用户ID）
        np.savez(feature_save_path, user_features=user_features, user_ids=user_ids_np)
        # 保存邻接表（字典）
        with open(adjacency_save_path, "w", encoding="utf-8") as f:
            json.dump(adjacency, f)

        return adjacency

    def _preprocess_graph(self):
        """构建PyG的图数据（适配原代码的GCN）：使用用户位置索引作为节点ID"""
        # 构建用户ID到位置索引的映射：{user_id: idx}
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        
        # 边索引（无向图，双向添加，转换为位置索引）
        edge_index = []
        for user_id, neighbors in self.adjacency.items():
            user_idx = user_id_to_idx[user_id]
            for neighbor in neighbors:
                neighbor_idx = user_id_to_idx[neighbor]
                edge_index.append([user_idx, neighbor_idx])
                edge_index.append([neighbor_idx, user_idx])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 节点特征：用行为序列作为初始特征
        x = torch.tensor(self.behaviors, dtype=torch.float32)
        # 节点标签
        y = torch.tensor(self.labels, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

    def _split_data(self):
        """分层抽样划分训练/测试集：返回用户的位置索引（而非用户ID）"""
        # 生成用户的位置索引（0,1,2...len(user_ids)-1）
        user_indices = list(range(len(self.user_ids)))
        # 分层抽样：按用户标签划分，返回位置索引的子集
        train_indices, test_indices = train_test_split(
            user_indices,
            train_size=self.config["data"]["train_ratio"],
            stratify=self.labels,  # self.labels是按位置索引顺序的，分层匹配
            random_state=42
        )
        # 转换为Tensor（整数类型）
        return torch.tensor(train_indices, dtype=torch.long), torch.tensor(test_indices, dtype=torch.long)


class UserProfileDataset(Dataset):
    """PyTorch数据集加载器：适配多模态数据（兼容list/Tensor索引）"""
    def __init__(self, dataset, indices, tokenizer, config):
        """
        参数说明：
        - dataset: SocialNetworkDataset实例
        - indices: 用户的位置索引（可以是list或Tensor类型）
        - tokenizer: BERT分词器
        - config: 配置字典
        """
        self.dataset = dataset
        # 核心修复：兼容list和Tensor类型的indices
        if isinstance(indices, torch.Tensor):
            self.indices = indices.tolist()  # Tensor转列表
        else:
            self.indices = indices  # 已是列表，直接赋值
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. 获取当前位置索引对应的用户全局索引
        user_global_idx = self.indices[idx]
        # 2. 根据全局索引获取用户数据
        user_id = self.dataset.user_ids[user_global_idx]
        text = self.dataset.texts[user_global_idx]
        behavior = self.dataset.behaviors[user_global_idx]
        label = self.dataset.labels[user_global_idx]
        
        # 3. BERT文本编码
        text_encoded = self.tokenizer(
            text,
            max_length=self.config["data"]["text_max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 4. 去除batch维度，返回数据
        return {
            "user_id": user_id,
            "input_ids": text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "behavior": torch.tensor(behavior, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """自定义Collate函数：批量处理数据"""
    return {
        "user_id": torch.tensor([item["user_id"] for item in batch], dtype=torch.long),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "behavior": torch.stack([item["behavior"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch])
    }

def get_data_loaders(dataset, config):
    """获取训练集/测试集的数据加载器"""
    train_dataset = UserProfileDataset(dataset, dataset.train_idx, dataset.tokenizer, config)
    test_dataset = UserProfileDataset(dataset, dataset.test_idx, dataset.tokenizer, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0  # Windows系统建议设为0，避免多进程问题
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0
    )
    return train_loader, test_loader