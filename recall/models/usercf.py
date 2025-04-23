import polars as pl
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import heapq
from pathlib import Path
import math
import pickle
from tqdm import tqdm
import numpy as np
from utils.exposure_data import get_all_expose_df
    

def get_item_user(expose_df: pl.DataFrame) -> Dict[int, List[int]]:
    """
    构建物品的用户点击字典

    根据用户点击日志生成结构字典，键为文章ID，值为点击该文章的用户ID列表。

    参数:
        expose_df (pl.DataFrame): 必须包含以下列:
            - user_id: 整数类型
            - article_id: 整数类型
            - is_clicked: 是否点击 (1 表示点击)

    返回:
        Dict[int, List[int]]: 物品的用户点击字典
            示例: {456: [123, 789]}
    """
    # 筛选点击记录
    filtered_df = expose_df.filter(pl.col("is_clicked") == 1)

    # 按 article_id 分组，获取对应的用户列表
    item_user_dict = (
        filtered_df
        .group_by("article_id")
        .agg(pl.col("user_id").alias("user_list"))
        .to_dict(as_series=False)  # 转换为 Python 字典
    )

    # 解析 Polars 结果，转换为标准 Python 格式
    return {
        article_id: user_list
        for article_id, user_list in zip(item_user_dict["article_id"], item_user_dict["user_list"])
    }


def usercf_sim(item_user_dict: Dict[int, List[Tuple[int, int]]], save_path: str, offline: bool = True) -> Dict[int, List[Tuple[int, float]]]:
    """
    计算用户之间的相似性矩阵（基于用户的协同过滤 UserCF），考虑用户活跃度和物品流行度。

    :param item_user_dict: Dict[int, List[int]]，物品的用户点击序列 {item_id: [user_id, ...]}
    :param save_path: str，相似性矩阵和倒排索引存储路径
    :param offline: bool, 是否为离线模式
    :return: Dict[int, List[Tuple[int, float]]]，每个用户的相似用户倒排索引表
             {
                user1: [(user2, 相似度值), (user3, 相似度值), ...],
                user2: [(user1, 相似度值), (user3, 相似度值), ...],
                ...
             }
    """
    u2u_sim: Dict[int, Dict[int, float]] = defaultdict(dict)  # 用户相似度矩阵
    user_item_cnt: Dict[int, int] = defaultdict(int)  # 记录每个用户的点击物品数量
        
    # 遍历每个物品的用户集合，构建用户共现矩阵
    for item, users in tqdm(item_user_dict.items(), desc="Building user co-occurrence matrix"):
        log_weight = 1.0 / math.log1p(len(users) + 1) # log(1 + |U_i|) 代表物品的流行度
        for u in users:
            user_item_cnt[u] += 1 # 统计每个用户点击的文章数量，用于计算用户的活跃度
            for v in users:
                if u == v:
                    continue
                
                u2u_sim[u].setdefault(v, 0)
                u2u_sim[u][v] += log_weight  # 计算相似度分子部分

    # 归一化相似度矩阵
    for u, related_users in tqdm(u2u_sim.items(), desc="Normalizing similarity matrix"):
        for v, sim_uv in related_users.items():
            u2u_sim[u][v] = sim_uv / math.sqrt(user_item_cnt[u] * user_item_cnt[v])

    # 构建倒排索引：每个用户的相似用户
    inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for u, related_users in tqdm(u2u_sim.items(), desc="Building user inverted index"):
        topk_users = heapq.nlargest(len(related_users), related_users.items(), key=lambda x: x[1])
        inverted_index[u] = [(v, sim) for v, sim in topk_users]

    # 存储相似性矩阵和倒排索引
    mode = "offline" if offline else "online"
    with open(f"{save_path}/usercf_u2u_sim_{mode}.pkl", "wb") as f:
        pickle.dump(u2u_sim, f)
    with open(f"{save_path}/usercf_inverted_index_{mode}.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    return inverted_index
    

if __name__ == '__main__':

    # 数据存储路径和保存路径
    offline_path = "/data3/zxh/news_rec/offline_data"
    online_path = "/data3/zxh/news_rec/online_data"
    save_path = '/data3/zxh/news_rec/temp_results'


    # 读取数据
    offline = True
    if offline:
        train_df, val_df, test_df = get_all_expose_df(offline_path, online_path,offline)
    else:
        train_df, test_df = get_all_expose_df(offline_path, online_path,offline)

    # 调用ItemCF召回通道进行新闻召回
    item_user_dict = get_item_user(train_df) # 物品的用户点击字典

    inverted_index = usercf_sim(item_user_dict, save_path, offline) # 计算用户之间的相似性矩阵