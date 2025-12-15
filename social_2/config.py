import os
import yaml
import random
import numpy as np
import torch
from loguru import logger

def load_config(config_path="config.yaml"):
    """加载配置文件，并强制转换参数类型"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 1. 创建文件夹（路径相关）
    for path in config["path"].values():
        os.makedirs(path, exist_ok=True)
    
    # 2. 强制转换数值类型（避免YAML解析为字符串）
    config["train"]["lr"] = float(config["train"]["lr"])
    config["train"]["batch_size"] = int(config["train"]["batch_size"])
    config["train"]["epochs"] = int(config["train"]["epochs"])
    config["train"]["patience"] = int(config["train"]["patience"])
    config["data"]["user_num"] = int(config["data"]["user_num"])
    config["data"]["train_ratio"] = float(config["data"]["train_ratio"])
    config["data"]["text_max_len"] = int(config["data"]["text_max_len"])
    config["data"]["behavior_seq_len"] = int(config["data"]["behavior_seq_len"])
    config["data"]["n_classes"] = int(config["data"]["n_classes"])
    
    # 3. 设备配置（自动适配CUDA/CPU）
    config["train"]["device"] = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    
    # 4. 日志配置
    logger.add(os.path.join(config["path"]["log_path"], "train.log"), rotation="500MB")
    
    return config

def set_seed(seed=42):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 新增：设置python内置的随机种子
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)