########################################################################
               #################模型训练##############
########################################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import time
from utils.config import load_configs
from data.preprocessing import build_feature_columns
from models.deepfm import DeepFM
from data.dataset import NewsDataset
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import datetime
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# 加载配置
config = load_configs()


# 加载特征列
feature_columns, feature_groups = build_feature_columns(config['feature'])


# 线性侧特征及交叉侧特征
linear_feature_columns_name = config['model']['deepfm_lhuc']['linear_feature_columns_name']
dnn_feature_columns_name = config['model']['deepfm_lhuc']['dnn_feature_columns_name']


linear_feature_columns  = [col for col in feature_columns if col.name in linear_feature_columns_name ]
dnn_feature_columns  = [col for col in feature_columns if col.name in dnn_feature_columns_name ]

# 初始化数据集处理器
dataset_creator = NewsDataset(config, feature_columns, model="deepfm")


# 创建训练数据集
dataset = {
    "train" : dataset_creator.create_dataset(data_path=config['training']['global']['train_data'], shuffle=True),
    "val" : dataset_creator.create_dataset(data_path=config['training']['global']['val_data'], shuffle=True),
    "test" : dataset_creator.create_dataset(data_path=config['training']['global']['test_data'], shuffle=True),
}

########################################################################
               #################模型训练##############
########################################################################
# 线上测试时需要拼接数据
full_dataset = dataset["train"].concatenate(dataset["val"])

model = DeepFM(linear_feature_columns, dnn_feature_columns,
               cate_map=config['feature']['categorical_vocabs'],
               fm_embed_dim=config['model']['deepfm']['fm_embed_dim'],
               dnn_hidden_units=config['model']['deepfm']['dnn_hidden_units'],
               dnn_activation=config['model']['deepfm']['dnn_activation'], 
               seed=config['model']['deepfm']['seed'],
               )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
               loss= "binary_crossentropy",  metrics=tf.keras.metrics.AUC(name='auc'))


history_loss = model.fit(full_dataset, epochs=1, verbose=1, validation_data=dataset['test'])

# 保存模型
model.save(f"/data3/zxh/news_rec/temp_results/deepfm/002/")  # 这会创建一个文件夹，包含 SavedModel 格式的模型




