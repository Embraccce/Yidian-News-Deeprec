########################################################################
               #################模型训练##############
########################################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
from utils.config import load_configs
from data.preprocessing import build_feature_columns
from models.esmm import ESMM
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


# 生成各组特征
linear_feature_columns = [col for col in feature_columns if col.name in config['model']['esmm']['linear_feature_columns_name'] ]
fm_cross_columns = [col for col in feature_columns if col.name in config['model']['esmm']['fm_cross_columns_name'] ]
dnn_feature_columns = [col for col in feature_columns if col.name in config['model']['esmm']['dnn_feature_columns_name'] ]


# 初始化数据集处理器
dataset_creator = NewsDataset(config, feature_columns, "essm")


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

# 模型的配置文件
esmm_config = config['model']['esmm']

model = ESMM(linear_feature_columns,fm_cross_columns,dnn_feature_columns,
             cate_map=config['feature']['categorical_vocabs'],
             fm_embed_dim=esmm_config['fm_embed_dim'],
             num_tasks=esmm_config['num_tasks'],
             tasks=esmm_config['tasks'],
             tasks_name=esmm_config['tasks_name'],
             num_experts=esmm_config['num_experts'],
             units_experts=esmm_config['units_experts'],
             task_dnn_units=esmm_config['task_dnn_units'],
             seed=esmm_config['seed'],
             dnn_activation=esmm_config['dnn_activation']
            )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss={"ctr": "binary_crossentropy",
                    "ctcvr": "binary_crossentropy",
                    },
              loss_weights=[1.0, 1.0],
              metrics={"ctr": [tf.keras.metrics.AUC(name='auc')],
                       "ctcvr": [tf.keras.metrics.AUC(name='auc')]}
              )



log_dir = os.path.join(
    config['training']['global']['log_dir'], 
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

history_loss = model.fit(full_dataset, epochs=1, verbose=1, validation_data=dataset['test'])

# # 保存模型
# model.save(f"/data3/zxh/news_rec/temp_results/essm/004/")  # 这会创建一个文件夹，包含 SavedModel 格式的模型


# # TensorBoard 使用TensorBoard记录信息
# tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
#                  histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  write_graph=True,  # 是否存储网络结构图
#                  write_images=True,# 是否可视化参数
#                  update_freq='epoch',
#                  embeddings_freq=0, 
#                  embeddings_layer_names=None, 
#                  embeddings_metadata=None,
#                         profile_batch = 20)

# history_loss = model.fit(dataset['train'], epochs=3, 
#           validation_data=dataset['val'],
#           verbose=1,callbacks=[tbCallBack])


# total_train_sample, total_val_sample = 159_283_939, 15_730_319
# val_batch_size = 4096
# train_steps_per_epoch = np.floor(total_train_sample/ config['training']['global']['batch_size']).astype(np.int32)
# test_steps_per_epoch = np.ceil(total_val_sample/ val_batch_size).astype(np.int32)

# history_loss = model.fit(dataset['train'], epochs=1, steps_per_epoch=train_steps_per_epoch,
#           validation_data=dataset['val'], validation_steps=test_steps_per_epoch,
#           verbose=1)



