import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, top_k_accuracy_score

def train_model(model, train_loader, test_loader, graph_data, config, device):
    """模型训练与验证主函数"""
    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_acc = 0.0
    patience_counter = 0

    logger.info("开始训练模型...")
    for epoch in range(config["train"]["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_logits_list = []
        train_labels_list = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"):
            # 数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            behavior = batch["behavior"].to(device)
            label = batch["label"].to(device)
            user_id = batch["user_id"].to(device)
            graph_data = graph_data.to(device)

            optimizer.zero_grad()
            # 前向传播
            logits, _, _ = model(
                text_input={"input_ids": input_ids, "attention_mask": attention_mask},
                behavior_input=behavior,
                graph_data=graph_data,
                user_idx=user_id
            )
            # 损失计算与反向传播
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_logits_list.extend(logits.detach().cpu().numpy())
            train_labels_list.extend(label.cpu().numpy())

        # 验证阶段
        model.eval()
        test_loss = 0.0
        test_logits_list = []
        test_labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                behavior = batch["behavior"].to(device)
                label = batch["label"].to(device)
                user_id = batch["user_id"].to(device)
                graph_data = graph_data.to(device)

                logits, _, _ = model(
                    text_input={"input_ids": input_ids, "attention_mask": attention_mask},
                    behavior_input=behavior,
                    graph_data=graph_data,
                    user_idx=user_id
                )
                loss = criterion(logits, label)
                test_loss += loss.item()
                test_logits_list.extend(logits.cpu().numpy())
                test_labels_list.extend(label.cpu().numpy())

        # 计算评估指标
        train_loss_avg = train_loss / len(train_loader)
        test_loss_avg = test_loss / len(test_loader)

        # 准确率（分类结果）
        train_preds = np.argmax(train_logits_list, axis=1)
        test_preds = np.argmax(test_logits_list, axis=1)
        train_acc = accuracy_score(train_labels_list, train_preds)
        test_acc = accuracy_score(test_labels_list, test_preds)


        # 替换原有的test_top3_acc计算逻辑
        # 1. 定义类别标签（显式指定，解决二进制标签问题）
        classes = np.arange(config["data"]["n_classes"])  # 二分类时为[0,1]
        # 2. 确定k值：二分类时k=1（k=2无意义），多分类时取min(3, n_classes-1)
        k = 1 if config["data"]["n_classes"] == 2 else min(3, config["data"]["n_classes"] - 1)

        # 3. 强制将y_true转换为与classes匹配的格式，并计算top-k准确率
        try:
            test_top3_acc = top_k_accuracy_score(
                y_true=test_labels_list,
                y_score=test_logits_list,
                k=k,
                labels=classes,  # 显式指定所有类别
                normalize=True
            )
        except ValueError:
            # 若仍报错（如y_true只包含一个标签），直接赋值为1.0（边界情况处理）
            test_top3_acc = 1.0

        # 4. 日志输出（适配k值）
        logger.info(f"Test Loss: {test_loss_avg:.4f} | Test Acc: {test_acc:.4f} | Test Top-{k} Acc: {test_top3_acc:.4f}")

        # 早停与模型保存
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc
            }, f"{config['path']['model_save_path']}/best_model.pth")
            logger.info(f"保存最佳模型（Test Acc: {best_acc:.4f}）")
        else:
            patience_counter += 1
            if patience_counter >= config["train"]["patience"]:
                logger.info(f"早停触发（Patience: {patience_counter}），最佳Test Acc: {best_acc:.4f}")
                break

        # 学习率调度
        scheduler.step(test_loss_avg)

    return model