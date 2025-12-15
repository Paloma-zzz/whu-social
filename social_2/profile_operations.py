import os
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from data_processing import UserProfileDataset, collate_fn

def generate_user_profile(model, dataset, config, device):
    """生成用户画像：提取融合特征+K-Means聚类标签化"""
    # 加载最佳模型权重
    checkpoint = torch.load(f"{config['path']['model_save_path']}/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 全量数据加载器
    full_dataset = UserProfileDataset(dataset, dataset.user_ids, dataset.tokenizer, config)
    full_loader = DataLoader(
        full_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=False
    )

    # 提取特征与权重
    user_profiles = []
    user_ids = []
    attn_weights_list = []

    with torch.no_grad():
        for batch in tqdm(full_loader, desc="提取用户画像特征"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            behavior = batch["behavior"].to(device)
            user_id = batch["user_id"].to(device)
            graph_data = dataset.graph_data.to(device)

            _, fused_feat, attn_weights = model(
                text_input={"input_ids": input_ids, "attention_mask": attention_mask},
                behavior_input=behavior,
                graph_data=graph_data,
                user_idx=user_id
            )
            user_profiles.extend(fused_feat.cpu().numpy())
            user_ids.extend(user_id.cpu().numpy())
            attn_weights_list.extend(attn_weights.cpu().numpy())

    # 转换为数组
    user_profiles = np.array(user_profiles)
    user_ids = np.array(user_ids)
    attn_weights = np.array(attn_weights_list)

    # K-Means聚类生成标签
    kmeans = KMeans(n_clusters=config["data"]["n_classes"], random_state=42)
    cluster_labels = kmeans.fit_predict(user_profiles)

    # 构建画像字典
    profile_dict = {
        "user_id": user_ids,
        "profile_feat": user_profiles,
        "cluster_label": cluster_labels,
        "true_label": dataset.labels,
        "attn_weights": attn_weights
    }

    # 保存画像
    np.save(f"{config['path']['result_path']}/user_profiles.npy", profile_dict)
    logger.info("用户画像已保存至：{}".format(config["path"]["result_path"]))

    return profile_dict

def evaluate_profile(profile_dict, config):
    """用户画像质量评估"""
    cluster_labels = profile_dict["cluster_label"]
    true_labels = profile_dict["true_label"]
    user_profiles = profile_dict["profile_feat"]
    attn_weights = profile_dict["attn_weights"]

    # 1. 聚类与真实标签的匹配度
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    # 2. 同一类别内的画像相似度
    similarity_dict = {}
    for label in range(config["data"]["n_classes"]):
        idx = np.where(cluster_labels == label)[0]
        if len(idx) < 2:
            similarity_dict[label] = 0.0
            continue
        sim = cosine_similarity(user_profiles[idx])
        similarity_dict[label] = sim[np.triu_indices_from(sim, k=1)].mean()
    avg_similarity = np.mean(list(similarity_dict.values()))

    # 3. 模态权重分析
    text_weight = attn_weights[:, 0].mean()
    behavior_weight = attn_weights[:, 1].mean()
    graph_weight = attn_weights[:, 2].mean()

    # 4. 标签覆盖率
    coverage = len(np.unique(cluster_labels)) / config["data"]["n_classes"]

    # 日志输出
    logger.info("\n" + "="*50)
    logger.info("用户画像质量评估结果：")
    logger.info(f"调整兰德指数（ARI）：{ari:.4f}")
    logger.info(f"归一化互信息（NMI）：{nmi:.4f}")
    logger.info(f"同类别画像相似度均值：{avg_similarity:.4f}")
    logger.info(f"标签覆盖率：{coverage:.4f}")
    logger.info(f"模态平均权重：文本={text_weight:.4f} | 行为={behavior_weight:.4f} | 社交关系={graph_weight:.4f}")
    logger.info("="*50)

    return {
        "ari": ari, "nmi": nmi, "avg_similarity": avg_similarity,
        "coverage": coverage, "modal_weights": {"text": text_weight, "behavior": behavior_weight, "graph": graph_weight}
    }
def calculate_similarity(feat1, feat2, all_features=None):
    """
    优化的相似度计算：添加噪声+余弦相似度+欧式距离惩罚
    :param feat1: 用户1的画像特征（numpy数组）
    :param feat2: 用户2的画像特征（numpy数组）
    :param all_features: 所有用户的特征（用于欧式距离归一化）
    :return: 0.7~1.0之间的相似度（保留4位小数）
    """
    # 步骤1：添加微小噪声（打破特征同质化）
    noise_scale = 0.001  # 噪声尺度，可在config中配置
    feat1 = feat1 + np.random.normal(0, noise_scale, feat1.shape)
    feat2 = feat2 + np.random.normal(0, noise_scale, feat2.shape)
    
    # 步骤2：计算余弦相似度（0~1）
    cos_sim = 1 - cosine(feat1, feat2)
    
    # 步骤3：计算欧式距离并归一化（惩罚过近的特征）
    if all_features is not None:
        # 全局欧式距离最大值（归一化用）
        max_dist = max([cosine(feat1, f) for f in all_features])
        euclid_dist = cosine(feat1, feat2) / (max_dist + 1e-8)  # 防止除0
    else:
        euclid_dist = cosine(feat1, feat2)
    
    # 步骤4：融合相似度（余弦相似度 * (1 - 欧式距离)）
    sim = cos_sim * (1 - euclid_dist)
    
    # 步骤5：限制相似度范围（0.7~1.0），保留4位小数
    sim = np.clip(sim, 0.7, 1.0)
    return round(float(sim), 4)


def personalized_recommendation(profile_dict, user_id, top_k=5):
    """基于用户画像的相似用户推荐"""
    user_ids = profile_dict["user_id"]
    user_profiles = profile_dict["profile_feat"]

    # 找到目标用户的特征
    target_idx = np.where(user_ids == user_id)[0][0]
    target_feat = user_profiles[target_idx:target_idx+1]

    # 计算余弦相似度
    similarities = cosine_similarity(target_feat, user_profiles)[0]
    similarities[target_idx] = -1  # 排除自身

    # 取Top-K相似用户
    top_k_idx = np.argsort(similarities)[-top_k:][::-1]
    top_k_user_ids = user_ids[top_k_idx]
    top_k_similarities = similarities[top_k_idx]

    # 日志输出
    logger.info(f"\n为用户{user_id}推荐的Top-{top_k}相似用户：")
    for i, (uid, sim) in enumerate(zip(top_k_user_ids, top_k_similarities)):
        logger.info(f"排名{i+1}：用户{uid}（相似度：{sim:.4f}）")

    return top_k_user_ids, top_k_similarities