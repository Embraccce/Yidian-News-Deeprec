########################################################################
               #################自定义Layer##############
########################################################################

from typing import Optional
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


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
    

class FMLayer(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        concated_embeds_value = inputs
        # 先求和再平方
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        # 先平方再求和
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
    

class MMoELayer(tf.keras.layers.Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, units_experts)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **units_experts**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, units_experts, num_experts, num_tasks,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MMoELayer, self).__init__(**kwargs)

        self.units_experts = units_experts
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer) or l2(l2=0.00001)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer) or l2(l2=0.00001)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(Dense(self.units_experts, activation=self.expert_activation,
                                                            use_bias=self.use_expert_bias,
                                                            kernel_initializer=self.expert_kernel_initializer,
                                                            bias_initializer=self.expert_bias_initializer,
                                                            kernel_regularizer=self.expert_kernel_regularizer,
                                                            bias_regularizer=self.expert_bias_regularizer,
                                                            activity_regularizer=self.activity_regularizer,
                                                            kernel_constraint=self.expert_kernel_constraint,
                                                            bias_constraint=self.expert_bias_constraint,
                                                            name=f'expert_net_{i}'))
        for i in range(self.num_tasks):
            self.gate_layers.append(Dense(self.num_experts, activation=self.gate_activation,
                                                          use_bias=self.use_gate_bias,
                                                          kernel_initializer=self.gate_kernel_initializer,
                                                          bias_initializer=self.gate_bias_initializer,
                                                          kernel_regularizer=self.gate_kernel_regularizer,
                                                          bias_regularizer=self.gate_bias_regularizer,
                                                          activity_regularizer=self.activity_regularizer,
                                                          kernel_constraint=self.gate_kernel_constraint,
                                                          bias_constraint=self.gate_bias_constraint,
                                                          name=f'gate_net_{i}'))

    def call(self, inputs, **kwargs):

        expert_outputs, gate_outputs, final_outputs = [], [], []

        # inputs: (batch_size, embedding_size)
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)

        # batch_size * units * num_experts
        expert_outputs = tf.concat(expert_outputs, 2)

        # [(batch_size, num_experts), ......]
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            # (batch_size, 1, num_experts)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)

            # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output,
                                                                                       self.units_experts, axis=1)

            # (batch_size, units)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # [(batch_size, units), ......]   size: num_task
        return final_outputs

    def get_config(self, ):
        config = {'units_experts': self.units_experts, 'num_experts': self.num_experts, 'num_tasks': self.num_tasks}
        base_config = super(MMoELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class DNN(tf.keras.layers.Layer):
    """The Multi Layer Perceptron with optional BatchNorm and Dropout.
      Input shape
        - nD tensor with shape: `(batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
      Output shape
        - nD tensor with shape: `(batch_size, ..., hidden_size[-1]). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, hidden_size[-1]).
      Arguments
        - **hidden_units**: list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **use_ln**: Whether to use LayerNormalization before activation.
        - **dropout_rates**: Tuple of float, dropout rates for each hidden layer (e.g., `(0.2, 0.3, 0.5)`).
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', use_bias=True, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, kernel_initializer='VarianceScaling',
                 kernel_regularizer=None, kernel_constraint=None,
                 activity_regularizer=None, seed=1024, l2_reg=1e-5,
                 dropout_rates=(0, 0, 0), use_ln=False,  # 新增参数
                 **kwargs):

        # Weight parameter
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer) or l2(l2=l2_reg)
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

        # Dropout & ln parameters
        self.dropout_rates = dropout_rates
        self.use_ln = use_ln

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn_layers = []  # 存储 Dense 层
        self.ln_layers = []   # 存储 LayerNorm 层（若启用）
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
            if self.use_ln:
                ln = LayerNormalization()
            else:
                ln = None
            self.ln_layers.append(ln)

            # 创建 LayerNorm 层（若启用）
            if self.use_ln:
                ln = LayerNormalization()
            else:
                ln = None
            self.ln_layers.append(ln)

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
            # 依次执行 Dense -> ln（可选） -> Dropout（可选）
            x = self.dnn_layers[i](dnn_input)
            if self.ln_layers[i] is not None:
                x = self.ln_layers[i](x)
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
    

class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    """

    def __init__(self, task='binary', **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class Cross(tf.keras.layers.Layer):
    """
    Cross Layer for DCN V2 (Deep & Cross Network v2)
    
    实现低秩分解形式的交叉网络层，用于显式特征交叉建模。
    计算公式：
        x_{l+1} = x0 * (V(U(x_l)) + diag_scale * x_l) + x_l
        其中 * 表示逐元素乘法。
    """

    def __init__(self, projection_dim: int, diag_scale: float = 0.0, use_bias: bool = True, **kwargs):
        """
        Args:
            projection_dim: U 和 V 矩阵的投影维度 r，通常 r << d 以降低计算成本。
            diag_scale: 用于训练稳定性的对角增强系数，非负数。
            use_bias: 是否在 V 层使用偏置项。
        """
        super(Cross, self).__init__(**kwargs)
        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias

    def build(self, input_shape):
        """
        初始化 U 和 V 两个小矩阵。
        输入维度为 (batch_size, d)，则：
            U: (d, r)
            V: (r, d)
        """
        last_dim = input_shape[-1]
        self._dense_u = tf.keras.layers.Dense(
            self._projection_dim, use_bias=False,
            kernel_initializer='glorot_uniform',
            name="cross_u"
        )
        self._dense_v = tf.keras.layers.Dense(
            last_dim, use_bias=self._use_bias,
            kernel_initializer='glorot_uniform',
            name="cross_v"
        )

    def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Args:
            x0: 初始输入张量 (batch_size, d)，通常是embedding后的原始特征。
            x: 上一层 Cross 的输出，若为 None 则默认等于 x0。
        Returns:
            x_{l+1}: 交叉后输出 (batch_size, d)
        """
        if x is None:
            x = x0  # 第一层时输入为 x0

        # 低秩映射 U(x) -> (batch_size, r)，再经过 V -> (batch_size, d)
        proj = self._dense_u(x)               # U(x): (B, r)
        proj_output = self._dense_v(proj)     # V(U(x)): (B, d)

        # 添加对角增强项，提升数值稳定性
        if self._diag_scale:
            proj_output += self._diag_scale * x

        # 特征交叉 + 残差连接
        return x0 * proj_output + x

    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_dim': self._projection_dim,
            'diag_scale': self._diag_scale,
            'use_bias': self._use_bias
        })
        return config
    

class LHUC(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_units=(512, 256, 128),
                 lhuc_size=64*6,  # 更清晰的参数命名
                 dnn_activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 kernel_initializer='VarianceScaling',
                 kernel_regularizer=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 name="LHUC",
                 l2_reg=1e-5,
                 **kwargs):
        """
        TensorFlow风格LHUC层
        :param hidden_units: 隐藏层单元数列表，如 (512, 256, 128)
        :param lhuc_size: LHUC输入特征维度
        :param activation: 激活函数
        :param dropout_rate: Dropout比率
        :param use_bn: 是否使用BatchNormalization
        :param kernel_initializer: 权重初始化方式
        """
        super(LHUC, self).__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.lhuc_size = lhuc_size
        self.dnn_activation = dnn_activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.l2_reg = l2_reg
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer) or l2(self.l2_reg)
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        # 主网络层
        self.dense_layers = []
        # LHUC缩放网络
        self.scale_generators = []
        # 正则化层
        self.bn_layers = []
        self.dropout_layer = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        current_dim = input_dim
        
        # 构建主网络
        for units in self.hidden_units:
            # LHUC缩放路径
            scale_gen = tf.keras.Sequential([
                Dense(256, activation='relu', kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,use_bias = self.use_bias,
                      bias_initializer = self.bias_initializer, name=f"{self.name}_scale_fc1_{current_dim}"),
                Dense(current_dim, activation='sigmoid',kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,use_bias = self.use_bias,
                      bias_initializer = self.bias_initializer, name=f"{self.name}_scale_fc2_{current_dim}")
            ], name=f"{self.name}_scale_gen_{current_dim}")
            self.scale_generators.append(scale_gen)

            # 主路径全连接
            dense = Dense(
                units=units, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,use_bias = self.use_bias,
                bias_initializer = self.bias_initializer, 
                name=f"{self.name}_dense_{units}"
            )
            self.dense_layers.append(dense)
            
            # 批量归一化
            if self.use_bn:
                self.bn_layers.append(BatchNormalization(name=f"{self.name}_bn_{current_dim}"))
            
            current_dim = units

        # Dropout层（所有层共享）
        if self.dropout_rate > 0:
            self.dropout_layer = Dropout(self.dropout_rate, name=f"{self.name}_dropout")
        
        super(LHUC, self).build(input_shape)

    def call(self, inputs, lhuc_input, training=None):
        """
        :param inputs: 主网络输入 [batch_size, input_dim]
        :param lhuc_input: LHUC控制输入 [batch_size, lhuc_size]
        :param training: 训练模式标志
        """
        x = inputs
        for i, (dense, scale_gen) in enumerate(zip(self.dense_layers, self.scale_generators)):
            scale = scale_gen(lhuc_input) * 2.0  # 生成缩放因子 [batch_size, current_dim], 缩放至0-2范围
            scaled_x = x * scale # 特征缩放
            x = dense(scaled_x) # 全连接
            if self.use_bn and i < len(self.bn_layers): # 批量归一化
                x = self.bn_layers[i](x, training=training)
            x = tf.keras.activations.get(self.dnn_activation)(x) # 激活函数
            if self.dropout_rate > 0: # Dropout
                x = self.dropout_layer(x, training=training)
        return x

    def get_config(self):
        config = super(LHUC, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'lhuc_size': self.lhuc_size,
            'dnn_activation': self.dnn_activation,
            'dropout_rate': self.dropout_rate,
            'use_batchnorm': self.use_bn,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer':self.kernel_regularizer,
            'use_bias':self.use_bias,
            'bias_initializer':self.bias_initializer,
            'l2_reg':self.l2_reg,
        })
        return config
    

#  AttentionPoolingLayer池化层   
class AttentionPoolingLayer(Layer):
    """
      Input shape
        - A list of three tensor: [query,keys,his_seq]
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - his_seq is a 2D tensor with shape: ``(batch_size, T)``
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **att_activation**: Activation function to use in attention net.
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **hist_mask_value**: the mask value of his_seq.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=True,
                 mode="sum",hist_mask_value=0, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.mode = mode
        self.hist_mask_value = hist_mask_value
        super(AttentionPoolingLayer, self).__init__(**kwargs)
        

    def build(self, input_shape):
        
        self.fc = tf.keras.Sequential()
        for unit in self.att_hidden_units:
            self.fc.add(Dense(unit, activation=self.att_activation, name="fc_att_"+str(unit))) 
        self.fc.add(Dense(1, activation=None, name="fc_att_out"))
        
        # 可训练的标量权重，用于 bias_emb 分数加权
        self.bias_proj = Dense(1, activation=None, name="bias_proj") # 这里指的是时间差

        super(AttentionPoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        query, keys, his_seq, bias_emb = inputs
        # 计算掩码
        key_masks = tf.not_equal(his_seq, tf.constant(self.hist_mask_value , dtype=his_seq.dtype))
        key_masks = tf.expand_dims(key_masks, 1)
        
        # 1. 转换query维度，变成历史维度T 
        # query是[B, 1,  H]，转换到 queries 维度为(B, T, H)，为了让pos_item和用户行为序列中每个元素计算权重, tf.shape(keys)[1] 结果就是 T
        queries = tf.tile(query, [1, tf.shape(keys)[1], 1]) # [B, T, H]

        # 2. 这部分目的就是为了在MLP之前多做一些捕获行为item和候选item之间关系的操作：加减乘除等。
        # 得到 Local Activation Unit 的输入。即 候选queries 对应的 emb，用户历史行为序列 keys
        # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后的结果
        din_all = tf.concat([queries, keys, queries-keys, queries*keys, bias_emb], axis=-1)
        # 3. attention操作，通过几层MLP获取权重，这个DNN 网络的输出节点为 1
        attention_score = self.fc(din_all) + self.bias_proj(bias_emb)  # [B, T, 1]
        # attention的输出, [B, 1, T]
        outputs = tf.transpose(attention_score, (0, 2, 1)) # [B, 1, T]

        # 4. 得到有真实意义的score
        if self.weight_normalization:
            # padding的mask后补一个很小的负数，这样后面计算 softmax 时, e^{x} 结果就约等于 0
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # 5. Activation，得到归一化后的权重
        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        # 6. 得到了正确的权重 outputs 以及用户历史行为序列 keys, 再进行矩阵相乘得到用户的兴趣表征
        # Weighted sum，
        if self.mode == 'sum':
            # outputs 的大小为 [B, 1, T], 表示每条历史行为的权重,
            # keys 为历史行为序列, 大小为 [B, T, H];
            # 两者用矩阵乘法做, 得到的结果 outputs 就是 [B, 1, H]
            outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        else:
            # 从 [B, 1, H] 变化成 Batch * Time
            outputs = tf.reshape(outputs, [-1, tf.shape(keys)[1]]) 
            # 先把scores在最后增加一维，然后进行哈达码积，[B, T, H] x [B, T, 1] =  [B, T, H]
            outputs = keys * tf.expand_dims(outputs, -1) 
            outputs = tf.reshape(outputs, tf.shape(keys)) # Batch * Time * Hidden Size
        
        return outputs
   
    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'mode': self.mode,}
        base_config = super(AttentionPoolingLayer, self).get_config()
        return config.update(base_config)
    
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 2
    seq_len = 5
    embedding_dim = 16
    bias_dim = 4  # bias_emb 的维度

    # 初始化测试输入
    query = tf.random.normal((batch_size, 1, embedding_dim))
    keys = tf.random.normal((batch_size, seq_len, embedding_dim))
    his_seq = tf.constant([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])  # 0 被设为 mask
    bias_emb = tf.random.normal((batch_size, seq_len, bias_dim))

    # 实例化 AttentionPoolingLayer
    att_layer = AttentionPoolingLayer(att_hidden_units=[16, 8], att_activation='relu')

    # 正常 forward 调用
    output = att_layer([query, keys, his_seq, bias_emb])

    # 打印输出维度，确认是否正确
    print("Output shape:", output.shape)  # 应为 [B, 1, H] 或 [B, T, H]，根据 mode 决定