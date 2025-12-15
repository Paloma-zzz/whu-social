import pandas as pd
import numpy as np

# 1. 读取原始数据集
df = pd.read_csv("weibo_senti_100k.csv")  # 你的数据集路径

# 2. 模拟1000个用户，给每条微博分配用户ID
num_users = 1000  # 可调整，建议1000以内（避免计算量过大）
df["user_id"] = np.random.randint(0, num_users, size=len(df))

# 3. 保存处理后的数据集（新增了user_id列）
df.to_csv("weibo_senti_100k_with_userid.csv", index=False)
print("已生成带用户ID的数据集：weibo_senti_100k_with_userid.csv")