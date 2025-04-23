import gc
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


def itemcf_sim(user_item_time_dict: Dict[int, List[Tuple[int, int]]], save_path: str, offline: bool = True, topk: int = 100) -> Dict[int, List[Tuple[int, float]]]:
    """
    计算文章与文章之间的相似性矩阵（基于物品的协同过滤 ItemCF）。

    :param user_item_time_dict: Dict[int, List[tuple[int, int]]]，用户点击的文章序列 {user_id: [(article_id, expose_time), ...]}
               - user_id (int): 用户 ID
               - article_id (int): 文章 ID
               - expose_time (float): 点击时间,这里已经筛选了点击序列
    :param save_path: str，相似性矩阵和倒排索引存储路径
    :param offline: bool, 是否为离线模式
    :param topk: int, 返回最相似的 topk 个物品
    :return: Dict[int, List[Tuple[int, float]]]，每个物品的相似物品倒排索引表
             格式：
             {
                item1: [(item2, 相似度值), (item3, 相似度值), ...],
                item2: [(item1, 相似度值), (item3, 相似度值), ...],
                ...
             }
    """
    # 文章相似度字典，存储文章共现关系
    i2i_sim: Dict[int, Dict[int, float]] = defaultdict(dict)
    item_cnt: Dict[int, int] = defaultdict(int)  # 记录每篇文章的点击次数

    # 遍历每个用户的点击序列，构建文章共现关系
    for user, item_time_list in tqdm(user_item_time_dict.items(), desc="Computing item co-occurrence"):
        item_count = len(item_time_list) # 当前用户点击的文章总数

        active_weight = 1 / math.log1p(item_count + 1)  # 计算 log(1 + |I_u|) 作为用户活跃度的权重

        for loc_1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1  # 统计每篇文章的点击次数，用于计算文章流行度

            for loc_2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue  # 跳过自身，避免自相似度计算

                # 计算文章 i 和 j 之间的位置信息影响 (方向性影响 c·β^(|l_i - l_j| - 1))
                loc_alpha = 1.0 if loc_2 > loc_1 else 0.7  # 正向浏览时 c=1.0，反向浏览时 c=0.7
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc_2 - loc_1) - 1))

                # 计算文章 i 和 j 之间的时间间隔影响 (exp(-α * |t_i - t_j|))
                time_weight = np.exp(-15000 * np.abs(i_click_time - j_click_time)) 
                
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += loc_weight * time_weight * active_weight  # 计算共现权重

    # 归一化相似性矩阵
    for i, related_items in tqdm(i2i_sim.items(), desc="Normalizing similarity matrix"):
        for j, wij in related_items.items():
            i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 归一化完成后立即释放中间变量
    del item_cnt, user_item_time_dict
    gc.collect()  # 强制垃圾回收


    # 构建倒排索引：每个物品的相似物品
    inverted_index: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for i, related_items in tqdm(i2i_sim.items(), desc="Building user inverted index"):
        # 使用堆维护相似物品的倒排索引pair
        topk_items = heapq.nlargest(topk, related_items.items(), key=lambda x: x[1])
        inverted_index[i] = [(j, sim) for j, sim in topk_items]

    # 存储相似性矩阵和倒排索引
    mode = "offline" if offline else "online"

    # 存储相似性矩阵和倒排索引
    # with open(f"{save_path}/itemcf_i2i_sim_{mode}.pkl", "wb") as f:
    #     pickle.dump(i2i_sim, f)

    with open(f"{save_path}/itemcf_top_{topk}_inverted_index_{mode}.pkl", "wb") as f:
        pickle.dump(inverted_index, f)

    return inverted_index
    

if __name__ == '__main__':
    # 数据存储路径和保存路径
    offline_path = "/home/ouc/data1/zxh/news_rec/offline_data"
    online_path = "/home/ouc/data1/zxh/news_rec/online_data"
    save_path = '/home/ouc/data1/zxh/news_rec/temp_results'
    # 读取数据
    offline = True
    if offline:
        train_df, val_df, test_df = get_all_expose_df(offline_path, online_path,offline)
    else:
        train_df, test_df = get_all_expose_df(offline_path, online_path,offline)

    # 调用ItemCF召回通道进行新闻召回
    user_item_time_dict = get_user_item_time(train_df) # 用户点击行为时间线字典

    inverted_index = itemcf_sim(user_item_time_dict, save_path, offline) # 构建物品相似物品的倒排索引

