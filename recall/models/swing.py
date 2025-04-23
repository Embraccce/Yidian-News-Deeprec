import polars as pl
from typing import Dict, Tuple, List
from collections import defaultdict
import heapq
from pathlib import Path
import math
import pickle
from tqdm import tqdm
import numpy as np
from utils.exposure_data import get_all_expose_df, get_user_item_time


def swing_sim(user_item_time_dict: Dict[int, List[Tuple[int, int]]], save_path: str, offline: bool = True, topk: int = 100) -> Dict[int, List[Tuple[int, float]]]:
    """
    计算基于 Swing 召回的物品相似性矩阵。

    :param user_item_time_dict: Dict[int, List[tuple[int, int]]]，用户点击的文章序列 {user_id: [(article_id, expose_time), ...]}
               - user_id (int): 用户 ID
               - article_id (int): 文章 ID
               - expose_time (float): 点击时间, 这里已经筛选了点击序列
    :param save_path: str，相似性矩阵和倒排索引存储路径
    :param offline: bool, 是否为离线模式
    :param topk: int, 返回最相似的 topk 个物品
    :return: Dict[int, List[Tuple[int, float]]]，每个物品的相似物品倒排索引表
             {
                item1: [(item2, 相似度值), (item3, 相似度值), ...],
                item2: [(item1, 相似度值), (item3, 相似度值), ...],
                ...
             }
    """

    # 1. 记录每个物品被哪些用户点击过，以及每个用户的点击记录
    item_user_cnt: Dict[int, int] = defaultdict(int)  # 记录每个物品的点击次数，即为 |U_i|
    user_co_items: Dict[Tuple[int, int], List[int]] = defaultdict(list)  # 记录用户组与共现物品的字典

    # 遍历用户点击序列，构建物品的共现关系
    for user, item_time_list in tqdm(user_item_time_dict.items(), desc="Building item-user interaction matrix"):
        for item, timestamp in item_time_list:
            item_user_cnt[item] += 1  # 统计物品点击次数

        # 计算每个用户的物品共现情况
        for idx1, (item1, time1) in enumerate(item_time_list):
            for idx2, (item2, time2) in enumerate(item_time_list):
                if item1 == item2:
                    continue
                
                key = (item1, item2) if item1 < item2 else (item2, item1)
                user_co_items[key].append(user)  # 记录用户组对应的物品

    # 2. 计算物品之间的 Swing 相似度
    item_sim: Dict[int, Dict[int, float]] = defaultdict()
    alpha = 5.0  # Swing 公式的平滑参数

    for (item1, item2), co_users in tqdm(user_co_items.items(), desc="Computing Swing similarity"):
        num_co_users = len(co_users)  # 共同点击该对物品的用户数，即|I_u ∩ I_v|
        overlap = 1.0 / (alpha + num_co_users)

        item_sim.setdefault(item1, {})
        item_sim.setdefault(item2, {})

        for user in co_users:            
            item_sim[item1][item2] = item_sim[item1].get(item2, 0.0) + overlap
            item_sim[item2][item1] = item_sim[item2].get(item1, 0.0) + overlap

    # 3. 归一化相似性矩阵
    for item1, related_items in tqdm(item_sim.items(), desc="Normalizing similarity matrix"):
        for item2, sim_value in related_items.items():
            item_sim[item1][item2] = sim_value / math.sqrt(item_user_cnt[item1] * item_user_cnt[item2])

    # 4. 构建倒排索引
    inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for item, related_items in tqdm(item_sim.items(), desc="Building user inverted index"):
        topk_items = heapq.nlargest(topk, related_items.items(), key=lambda x: x[1])
        inverted_index[item] = [(j, sim) for j, sim in topk_items]

    # 5. 存储计算结果
    mode = "offline" if offline else "online"
    # with open(f"{save_path}/swing_i2i_sim_{mode}.pkl", "wb") as f:
    #     pickle.dump(item_sim, f)
    
    with open(f"{save_path}/swing_top_{topk}_inverted_index_{mode}.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    return inverted_index
    

if __name__ == '__main__':
    # 数据存储路径和保存路径
    offline_path = "/data3/zxh/news_rec/offline_data"
    online_path = "/data3/zxh/news_rec/online_data"
    save_path = '/data3/zxh/news_rec/temp_results'

    # 读取数据
    offline = False
    if offline:
        train_df, val_df, test_df = get_all_expose_df(offline_path, online_path,offline)
    else:
        train_df, test_df = get_all_expose_df(offline_path, online_path, offline)

    # 调用Swing召回通道进行新闻召回
    user_item_time_dict = get_user_item_time(train_df) # 用户点击行为时间线字典

    inverted_index = swing_sim(user_item_time_dict, save_path, offline) # 构建物品相似物品的倒排索引