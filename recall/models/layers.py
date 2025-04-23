########################################################################
               #################自定义Layer##############
########################################################################

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *


class VocabLayer(tf.keras.layers.Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (0) # mask成 0
            idx = tf.where(masks, idx, paddings)
        return idx
    
    def get_config(self):  
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


class EmbeddingLookupSparse(tf.keras.layers.Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum',**kwargs):
        
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding
    
    
    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)
        
    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding,sp_ids=idx, sp_weights=val, combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding,sp_ids=idx, sp_weights=None, combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)
    
    def get_config(self):  
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner':self.combiner})
        return config


class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, embedding, **kwargs):
        
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding
    
    
    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)
        
    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed
    
    def get_config(self):  
        config = super(EmbeddingLookup, self).get_config()
        return config

    

# 稠密转稀疏 
class DenseToSparseTensor(tf.keras.layers.Layer):
    def __init__(self, mask_value= -1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value
        

    def call(self, dense_tensor):    
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value , dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor
    
    def get_config(self):  
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config
    

class HashLayer(tf.keras.layers.Layer):
    """
    Hash the input to [0, num_buckets)
    - 支持 int 和 string 类型输入
    - 当 mask_zero=True 时：
      * 对于数值类型：0 会被映射为 0，其他值哈希到 [1, num_buckets)
      * 对于字符串类型：字符串 "0" 会被映射为 0，其他值哈希到 [1, num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HashLayer, self).build(input_shape)

    def call(self, x, **kwargs):

        if x.dtype in (tf.int32, tf.int64):
            x = tf.as_string(x)
        
        # 计算哈希桶数
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        
        # 哈希核心逻辑
        if self.mask_zero:
            # 创建掩码（识别0值）
            zero_mask = tf.cast(tf.not_equal(x, "0"), tf.int64)
            # 执行哈希并应用掩码
            hash_values = tf.strings.to_hash_bucket_fast(x, num_buckets) + 1
            return hash_values * zero_mask
        else:
            return tf.strings.to_hash_bucket_fast(x, self.num_buckets)

    def get_config(self):
        config = super(HashLayer, self).get_config()
        config.update({
            'num_buckets': self.num_buckets,
            'mask_zero': self.mask_zero
        })
        return config
    

class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)
    

class DNN(tf.keras.layers.Layer):
    """The Multi Layer Perceptron with optional BatchNorm and Dropout.
      Input shape
        - nD tensor with shape: `(batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
      Output shape
        - nD tensor with shape: `(batch_size, ..., hidden_size[-1]). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, hidden_size[-1]).
      Arguments
        - **hidden_units**: list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **use_bn**: Whether to use BatchNormalization before activation.
        - **dropout_rates**: Tuple of float, dropout rates for each hidden layer (e.g., `(0.2, 0.3, 0.5)`).
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', use_bias=True, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, kernel_initializer='VarianceScaling',
                 kernel_regularizer=None, kernel_constraint=None,
                 activity_regularizer=None, seed=1024, 
                 dropout_rates=(0, 0, 0), use_bn=False,  # 新增参数
                 **kwargs):

        # Weight parameter
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        # Activation parameter
        self.activation = activation

        # Bias parameter
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        # hidden_units parameter
        self.hidden_units = hidden_units
        self.seed = seed

        # Dropout & BN parameters
        self.dropout_rates = dropout_rates
        self.use_bn = use_bn

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn_layers = []  # 存储 Dense 层
        self.bn_layers = []   # 存储 BatchNorm 层（若启用）
        self.dropout_layers = []  # 存储 Dropout 层（若启用）
        for i in range(len(self.hidden_units)):
            # 创建 Dense 层
            dense = Dense(
                units=self.hidden_units[i],
                activation=self.activation,  # 所有层使用相同激活函数
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
            )
            self.dnn_layers.append(dense)
            # 创建 BatchNorm 层（若启用）
            if self.use_bn:
                bn = BatchNormalization()
            else:
                bn = None
            self.bn_layers.append(bn)

            # 创建 BatchNorm 层（若启用）
            if self.use_bn:
                bn = BatchNormalization()
            else:
                bn = None
            self.bn_layers.append(bn)

            # 创建 Dropout 层（若 rate > 0）
            dropout_rate = self.dropout_rates[i] if i < len(self.dropout_rates) else 0
            if dropout_rate > 0:
                dropout = Dropout(rate=dropout_rate, seed=self.seed)
            else:
                dropout = None
            self.dropout_layers.append(dropout)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        dnn_input = inputs

        for i in range(len(self.hidden_units)):
            # 依次执行 Dense -> BN（可选） -> Dropout（可选）
            x = self.dnn_layers[i](dnn_input)
            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x, training=training)  # 传递 training 状态给 BN
            if self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)  # 传递 training 状态给 Dropout
            dnn_input = x  # 更新输入为当前层输出

        return dnn_input

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units, 'seed': self.seed,
                  'use_bias': self.use_bias, 'kernel_initializer': self.kernel_initializer,
                  'bias_initializer': self.bias_initializer, 'kernel_regularizer': self.kernel_regularizer,
                  'bias_regularizer': self.bias_regularizer, 'activity_regularizer': self.activity_regularizer,
                  'kernel_constraint': self.kernel_constraint, 'bias_constraint': self.bias_constraint, }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

# Batch内负采样
class InBatchSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        # 制作概率修正查找表
        total_count = np.sum(list(self.sampler_config['item_count'].values()))
        item_prob_dict = {k: v / total_count for k, v in self.sampler_config['item_count'].items()}

        # 创建 TensorFlow 查找表
        ids = tf.constant(list(item_prob_dict.keys()), dtype=tf.string)
        counts = tf.constant(list(item_prob_dict.values()), dtype=tf.float32)
        self.item_count_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(ids, counts),
            default_value=0.0  # 如果 item_idx 中出现未在字典中的 ID，则返回 0
        )

        super(InBatchSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InBatchSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        user_vec, item_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.string:
            item_idx = tf.cast(item_idx, tf.string)
        user_vec /= self.temperature
        logits = tf.matmul(user_vec, item_vec, transpose_b=True)
        loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count_table, item_idx)
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
        base_config = super(InBatchSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

# 采样概率修正
def inbatch_softmax_cross_entropy_with_logits(logits, item_count_table, item_idx):
    # 根据 item_idx 查找概率值
    Q = item_count_table.lookup(tf.squeeze(item_idx, axis=1))

    try:
        logQ = tf.reshape(tf.math.log(Q), (1, -1))
        logits -= logQ  # subtract_log_q
        labels = tf.linalg.diag(tf.ones_like(logits[0]))
    except AttributeError:
        logQ = tf.reshape(tf.log(Q), (1, -1))
        logits -= logQ  # subtract_log_q
        labels = tf.diag(tf.ones_like(logits[0]))

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return loss