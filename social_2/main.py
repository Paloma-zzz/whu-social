from config import load_config, set_seed
from data_processing import SocialNetworkDataset, get_data_loaders
from models import MultiModalUserProfile
from train_eval import train_model
from profile_operations import generate_user_profile, evaluate_profile, personalized_recommendation
from loguru import logger

def main():
    # 1. 初始化配置与随机种子
    config = load_config()
    set_seed(42)

    # 2. 生成并预处理数据集
    dataset = SocialNetworkDataset(config)
    logger.info(f"数据集生成完成，用户数量：{dataset.user_num}")
    train_loader, test_loader = get_data_loaders(dataset, config)

    # 3. 初始化模型
    model = MultiModalUserProfile(config).to(config["train"]["device"])
    logger.info(f"模型初始化完成，设备：{config['train']['device']}")

    # 4. 训练模型
    model = train_model(model, train_loader, test_loader, dataset.graph_data, config, config["train"]["device"])

    # 5. 生成用户画像
    profile_dict = generate_user_profile(model, dataset, config, config["train"]["device"])

    # 6. 评估画像质量
    eval_results = evaluate_profile(profile_dict, config)

    # 7. 个性化推荐示例（第一个用户）
    target_user_id = dataset.user_ids[0]
    personalized_recommendation(profile_dict, target_user_id, top_k=5)

if __name__ == "__main__":
    main()