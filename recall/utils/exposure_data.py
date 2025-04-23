import polars as pl
from typing import Dict, List, Tuple

def get_all_expose_df(offline_path="/data3/zxh/news_rec/offline_data", 
                     online_path="/data3/zxh/news_rec/online_data",
                     offline=True):
    """
    获取曝光数据集
    
    参数：
    - offline_path: 离线训练数据路径
    - online_path: 在线测试数据路径
    - offline: 是否为离线训练模式，若为 False，则合并训练集和验证集
    
    返回：
    - train_df: 训练数据集
    - test_df: 测试数据集
    """
    # 读取训练和验证数据
    train_df = pl.read_ipc(f"{offline_path}/train_data_offline.ipc")
    val_df = pl.read_ipc(f"{offline_path}/val_data_offline.ipc")
    test_df = pl.read_ipc(f"{online_path}/test_data_online.ipc")

    # 处理 offline/online 逻辑
    if offline:
        return train_df, val_df, test_df
    else:
        train_df = pl.concat([train_df, val_df], how="vertical")
        return train_df, test_df

def get_user_item_time(expose_df: pl.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    """
    构建用户点击行为时间线字典

    根据用户点击日志生成结构字典，键为用户ID，值为按时间排序的(文章ID, 时间戳)元组列表
    
    参数:
        expose_df (pl.DataFrame): 必须包含以下列:
            - user_id: 整数类型
            - article_id: 整数类型  
            - expose_time: float（时间戳, ms）
    
    返回:
        Dict[int, List[Tuple[int, int]]]: 用户点击时间线字典
            示例: {123: [(456, 0.162), (789, 0.179)]}
    """
    # 筛选点击数据，并按时间戳排序
    sorted_df = expose_df.filter(pl.col("is_clicked") == 1).sort("expose_time")

    # 计算 expose_time 归一化
    min_value, max_value = sorted_df["expose_time"].min(), sorted_df["expose_time"].max()
    sorted_df = sorted_df.with_columns(
        ((pl.col("expose_time") - min_value) / (max_value - min_value)).alias("expose_time")
    )

    # 按 user_id 分组，构造 (article_id, expose_time) 组合
    user_item_time_dict = (
        sorted_df
        .group_by("user_id")
        .agg(pl.struct("article_id", "expose_time").alias("click_list"))
        .to_dict(as_series=False)
    )

    return {
        user_id: [(entry["article_id"], entry["expose_time"]) for entry in click_list]
        for user_id, click_list in zip(user_item_time_dict["user_id"], user_item_time_dict["click_list"])
    }
