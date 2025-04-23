########################################################################
               #################自定义dataset##############
########################################################################
import numpy as np
import tensorflow as tf
import polars as pl
from collections import namedtuple, OrderedDict
from typing import Dict, Tuple
from utils.config import SparseFeat, DenseFeat, VarLenSparseFeat

class NewsDataset:
    def __init__(self, config: Dict, feature_columns):
        self.feature_cfg = config['feature']
        self.train_cfg = config['training']
        self.feature_columns = feature_columns
        # 初始化预训练嵌入
        self.item_id2idx, self.item_title_embedding = self._load_pretrained_embeddings(self.train_cfg['global']['embedding_data'])
        # 获取csv的列名和填充值
        self.col_name = self.feature_cfg['csv_schema']
        self.default_values = self._get_default_values(self.col_name, self.feature_columns)
        # 获取填充形状和填充值
        self.pad_shapes, self.pad_values = self._get_pad_shapes_and_values(self.feature_columns)
        # 需要分桶的参数
        self.bucket_dict = self.feature_cfg.get('bucket_config')


    def _load_pretrained_embeddings(self, ipc_file) -> Tuple[tf.lookup.StaticHashTable, tf.Tensor]:
        """加载预训练标题嵌入"""

        df = pl.read_ipc(ipc_file)
        # 获取所有 article_id
        item_bert_id = df["article_id"].cast(pl.Utf8).to_list()

        # 获取所有嵌入
        item_bert_embed = df.drop("article_id").to_numpy()

        # 构建 item_id2idx 查找表
        item_id2idx = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=item_bert_id, 
                values=range(1, len(item_bert_id) + 1), 
                key_dtype=tf.string, 
                value_dtype=tf.int32
            ),
            default_value=0
        )

        # 添加一个全零向量，索引 0 代表未知 ID
        item_bert_embed = np.vstack([[0.0] * 32, item_bert_embed])
        # 转换为 TensorFlow 张量
        item_title_embedding = tf.constant(item_bert_embed, dtype=tf.float32)

        return item_id2idx, item_title_embedding
    

    def _get_default_values(self, col_name, feature_columns) -> list:
        """根据 feature_columns 获取默认值"""
        default_values = []

        # 创建两个字典，快速查找特征的 dtype 和类型
        feature_dtype_map = {feat.name: feat.dtype for feat in feature_columns}
        feature_type_map = {feat.name: type(feat) for feat in feature_columns}  # 记录特征类型

        for feature_name in col_name:
            dtype = feature_dtype_map.get(feature_name)  # 默认使用 float32
            feat_type = feature_type_map.get(feature_name)  # 获取特征类型

            if feat_type == VarLenSparseFeat:
                default_values.append(["-1:1"]) # -1为未知类,这样既可以兼容string也可以兼容int,权重设置为1
            elif dtype in ['float32', 'float64']:
                default_values.append([0.0])
            elif dtype == 'string':
                default_values.append(["0"])
            elif dtype == 'int64':
                default_values.append(tf.constant([0], dtype=tf.int64))
            else:
                default_values.append([0])

        return default_values

    def _get_pad_shapes_and_values(self, feature_columns) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        生成特征的填充形状 (pad_shapes) 和填充值 (pad_values)
        
        Args:
            feature_columns (list): 包含特征列的列表
        
        Returns:
            tuple: (pad_shapes, pad_values)
        """
        pad_shapes = {}
        pad_values = {}

        for feat_col in feature_columns:
            if isinstance(feat_col, VarLenSparseFeat):
                max_tokens = feat_col.maxlen
                pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
                pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
                if feat_col.weight_name is not None:
                    pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
                    pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)

        # no need to pad labels 
            elif isinstance(feat_col, SparseFeat):
                pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
                pad_shapes[feat_col.name] = tf.TensorShape([])
            elif isinstance(feat_col, DenseFeat):
                if not feat_col.pre_embed:
                    pad_shapes[feat_col.name] = tf.TensorShape([])
                else:
                    pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])

                if feat_col.dtype == 'float32':
                    pad_values[feat_col.name] = 0.0
                elif feat_col.dtype == 'int32':
                    pad_values[feat_col.name] = 0
                else:
                    pad_values[feat_col.name] = tf.constant(0, dtype=tf.int64)

        pad_shapes = (pad_shapes, (tf.TensorShape([])))
        pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

        return pad_shapes, pad_values

    def _parse_function(self, example_proto: tf.Tensor) -> Tuple[Dict, tf.Tensor]:
        item_feats = tf.io.decode_csv(example_proto, record_defaults=self.default_values, field_delim='\t')
        parsed = dict(zip(self.col_name, item_feats))
        
        feature_dict = {}
        for feat_col in self.feature_columns:
            if isinstance(feat_col, VarLenSparseFeat):
                # 公共处理部分：分割原始特征值
                split_values = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                
                if feat_col.weight_name:
                    # 带权重的处理
                    kv_pairs = tf.strings.split(split_values, ':').to_tensor()
                    feat_ids, feat_vals = tf.unstack(tf.reshape(kv_pairs, [-1, 2]), axis=1)
                    
                    # 类型转换
                    if feat_col.dtype != 'string':
                        feat_ids = tf.strings.to_number(feat_ids, tf.int32)
                    feat_vals = tf.strings.to_number(feat_vals, tf.float32)
                    
                    # 存储结果
                    feature_dict.update({
                        feat_col.name: feat_ids,
                        feat_col.weight_name: feat_vals
                    })
                else:
                    # 不带权重的处理
                    feat_ids = tf.reshape(split_values, [-1])
                    if feat_col.dtype != 'string':
                        feat_ids = tf.strings.to_number(feat_ids, tf.int32)
                    feature_dict[feat_col.name] = feat_ids
        
            elif isinstance(feat_col, SparseFeat):
                feature_dict[feat_col.name] = parsed[feat_col.name]
                
            elif isinstance(feat_col, DenseFeat):
                if not feat_col.pre_embed:
                    feature_dict[feat_col.name] = parsed[feat_col.name]
                elif feat_col.reduce_type is not None: 
                    keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                    emb = tf.nn.embedding_lookup(params=self.item_title_embedding, ids=self.item_id2idx.lookup(keys))
                    emb = tf.reduce_mean(emb,axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                else:
                    emb = tf.nn.embedding_lookup(params=self.item_title_embedding, ids=self.item_id2idx.lookup(parsed[feat_col.pre_embed]))                
                    feature_dict[feat_col.name] = emb
            else:
                raise "unknown feature_columns...."

        # 分桶离散化
        if self.bucket_dict is not None:
            for ft in self.bucket_dict:
                feature_dict[ft] = tf.raw_ops.Bucketize(
                    input=feature_dict[ft],
                    boundaries=self.bucket_dict[ft])
                
        label = 1 # 都是正样本
        return feature_dict, label

    def create_dataset(self, data_path: str, shuffle: bool = False) -> tf.data.Dataset:
        """创建数据管道"""
        # 构建数据集
        dataset = tf.data.Dataset.list_files(data_path)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False if shuffle else True
        )
        
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.train_cfg['global'].get('shuffle_buffer_size', 10000),
                reshuffle_each_iteration=True
            )
            
        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 动态批处理
        dataset = dataset.padded_batch(
            batch_size=self.train_cfg['global']['batch_size'],
            padded_shapes=self.pad_shapes,
            padding_values=self.pad_values
        )
        
        # 性能优化
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
