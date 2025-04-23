########################################################################
               #################模型训练##############
########################################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel('ERROR')  # 设置日志级别为ERROR
import pickle
import time
from utils.config import load_configs, NegativeSampler
from data.preprocessing import build_feature_columns
from models.dssm import DSSM
from data.dataset import NewsDataset
from tensorflow.python.keras import backend as K
import datetime
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# 设置mode
mode = "online"

# 加载配置
config = load_configs()


# 加载特征列
feature_columns, feature_groups = build_feature_columns(config['feature'])


# 加载物料热度
with open(config['training']['global'][f"item_count_{mode}_dict"], "rb") as f:
    item_count_dict = pickle.load(f)
sampler_config = NegativeSampler(sampler='inbatch', num_sampled=None, item_name="article_id", item_count=item_count_dict, distortion=None)


# 初始化数据集处理器
dataset_creator = NewsDataset(config, feature_columns)


# 创建训练数据集
dataset = {
    "train" : dataset_creator.create_dataset(data_path=config['training']['global']['train_data'], shuffle=True),
    "val" : dataset_creator.create_dataset(data_path=config['training']['global']['val_data'], shuffle=True)
}

########################################################################
               #################模型训练##############
########################################################################
# 加载DSSM配置文件
dssm_config = config['model']['dssm']

model = DSSM(
    user_feature_columns=feature_groups['user'],  # 需要提供用户特征列
    item_feature_columns=feature_groups['item'],  # 需要提供物品特征列
    cate_map=config['feature']['categorical_vocabs'],
    **dssm_config,  # 解包字典，将 YAML 配置的参数传入
    sampler_config=sampler_config
)

model.compile(optimizer='adam', loss=lambda y_true, y_pred: K.mean(y_pred))

# log_dir = os.path.join(
#     config['training']['global']['log_dir'], 
#     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# )

# # # TensorBoard 使用TensorBoard记录信息
# # tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
# #                  histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
# #                  write_graph=True,  # 是否存储网络结构图
# #                  write_images=True,# 是否可视化参数
# #                  update_freq='epoch',
# #                  embeddings_freq=0, 
# #                  embeddings_layer_names=None, 
# #                  embeddings_metadata=None,
# #                         profile_batch = 20)


# total_train_sample, total_val_sample = 22_747_279, 2_183_545
full_dataset = dataset["train"].concatenate(dataset["val"])
# train_steps_per_epoch = np.floor(total_train_sample/ config['training']['global']['batch_size']).astype(np.int32)
# test_steps_per_epoch = np.ceil(total_val_sample/ val_batch_size).astype(np.int32)

# history_loss = model.fit(dataset['train'], epochs=3, steps_per_epoch=train_steps_per_epoch,
#           validation_data=dataset['val'], validation_steps=test_steps_per_epoch,
#           verbose=1)

history_loss = model.fit(full_dataset, epochs=5, verbose=1)

# 用户塔 item塔定义
user_embedding_model = tf.keras.Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = tf.keras.Model(inputs=model.item_input, outputs=model.item_embedding)

# 保存模型
tf.keras.models.save_model(user_embedding_model,f"/data3/zxh/news_rec/temp_results/dssm_user/003_{mode}/")
tf.keras.models.save_model(item_embedding_model,f"/data3/zxh/news_rec/temp_results/dssm_item/003_{mode}/")