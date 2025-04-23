from collections import OrderedDict, namedtuple
from models.layers import *
from tensorflow.keras.layers import *
from utils.config import SparseFeat, VarLenSparseFeat, DenseFeat


########################################################################
        #################定义生成特征分组和特征列##############
########################################################################
def build_feature_columns(feature_cfg):
    """根据合并后的配置生成特征分组和特征列"""
    feature_columns = []

    # 处理稀疏特征
    for feat in feature_cfg['feature_types']['sparse']:
        feature_columns.append(
            SparseFeat(
                name=feat['name'],
                voc_size=feat['voc_size'],
                hash_size=feat.get('hash_size', None),  # 可选参数
                share_embed=feat.get('share_embed', None),  # 根据实际情况配置
                embed_dim=feat['embed_dim'],
                dtype=feat['dtype']
            )
        )

    # 处理稠密特征
    for feat in feature_cfg['feature_types']['dense']:
        feature_columns.append(
            DenseFeat(
                name=feat['name'],
                pre_embed=feat.get('pre_embed', None),
                reduce_type=feat.get('reduce_type', None),
                dim=feat['dim'],
                dtype=feat['dtype']
            )
        )

    # 处理变长稀疏特征
    for feat in feature_cfg['feature_types']['varlen']:
        feature_columns.append(
            VarLenSparseFeat(
                name=feat['name'],
                voc_size=feat['voc_size'],
                hash_size=feat.get('hash_size', None),
                share_embed=None,  # 根据实际情况配置
                weight_name=feat.get('weight_name', None),
                combiner=feat['combiner'],
                embed_dim=feat['embed_dim'],
                maxlen=feat['maxlen'],
                dtype=feat['dtype']
            )
        )

    # 添加特征分组信息（可选）
    feature_groups = {
        'user': [col for col in feature_columns if col.name in feature_cfg['feature_groups']['user_features']],
        'item': [col for col in feature_columns if col.name in feature_cfg['feature_groups']['item_features']],
        'scene': [col for col in feature_columns if col.name in feature_cfg['feature_groups']['scene_features']],
        'cross': [col for col in feature_columns if col.name in feature_cfg['feature_groups']['cross_features']]
    }

    return feature_columns, feature_groups


########################################################################
               #################定义输入帮助函数##############
########################################################################

# 定义model输入特征
def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:    
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)         
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构造 自定义embedding层 matrix
def build_embedding_matrix(features_columns, dim=None, name_suf=None):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, (SparseFeat, VarLenSparseFeat)):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            vocab_size = feat_col.voc_size + 2
            embed_dim = feat_col.embed_dim if dim is None else dim
            if dim is None:
                name_tag = '_dnn'
            elif dim == 1:
                name_tag = '_linear'
            else:
                name_tag = '_fm'
            if name_suf is not None:
                name_tag = name_suf
            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim),mean=0.0, 
                                                                           stddev=0.001, dtype=tf.float32), trainable=True, name=f'{vocab_name}_embed{name_tag}')
    return embedding_matrix


def build_embedding_dict(features_columns, dim=None, name_suf=None):
    """构建自定义embedding层字典，支持普通嵌入和线性嵌入两种模式
    
    Args:
        features_columns: 特征列配置
        dim: 嵌入的维度，默认为None，表示按照特征来设定
        
    Returns:
        dict: 包含各特征对应embedding层的字典
    """
    if dim is None:
        name_tag = '_dnn'
    elif dim == 1:
        name_tag = '_linear'
    else:
        name_tag = '_fm'
    if name_suf is not None:
        name_tag = name_suf
    embedding_matrix = build_embedding_matrix(features_columns, dim=dim, name_suf=name_suf)
    embedding_dict = {}

    for feat_col in features_columns:
        # 统一处理共享embedding名称
        vocab_name = getattr(feat_col, 'share_embed', None) or feat_col.name
        
        # 稀疏特征处理
        if isinstance(feat_col, SparseFeat):
            embedding_dict[feat_col.name] = EmbeddingLookup(
                embedding=embedding_matrix[vocab_name],
                name=f'emb_lookup_{feat_col.name}{name_tag}'
            )
        
        # 变长稀疏特征处理
        elif isinstance(feat_col, VarLenSparseFeat):
            # 带有聚合方式的处理
            if feat_col.combiner is not None:
                sparse_args = {
                    'embedding': embedding_matrix[vocab_name],
                    'combiner': feat_col.combiner,
                    'name': f'emb_lookup_sparse_{feat_col.name}{name_tag}'
                }
                if feat_col.weight_name is not None:
                    sparse_args['has_weight'] = True
                
                embedding_dict[feat_col.name] = EmbeddingLookupSparse(**sparse_args)
            
            # 无聚合方式的常规处理
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(
                    embedding=embedding_matrix[vocab_name],
                    name=f'emb_lookup_{feat_col.name}{name_tag}'
                )

    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict, cate_map):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            _input = features[feat_col.name]
            if feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)
            else:
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                if vocab_name in cate_map:
                    keys = cate_map[vocab_name]
                    _input = VocabLayer(keys)(_input)

            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            _input = features[feat_col.name]
            if feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)
            else:
                mask_val = '0' if feat_col.dtype == 'string' else 0
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                if vocab_name in cate_map:
                    keys = cate_map[vocab_name]
                    _input = VocabLayer(keys, mask_value=mask_val)(_input)
            if feat_col.combiner is not None:
                input_sparse = DenseToSparseTensor(mask_value=0)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)

            sparse_embedding_list.append(embed)

        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])

        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)
    
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise "dnn_feature_columns can not be empty list"

        
def get_linear_logit(sparse_embedding_list, dense_value_list):
    
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add()([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        return dense_linear_layer
    else:
        raise "linear_feature_columns can not be empty list"

