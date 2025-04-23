import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from data.preprocessing import *

########################################################################
               #################定义模型##############
########################################################################
def DSSM(
    user_feature_columns,
    item_feature_columns,
    cate_map,
    user_dnn_hidden_units=(320, 200, 80),
    item_dnn_hidden_units=(320, 200, 80),
    user_dnn_dropout=(0, 0, 0),
    item_dnn_dropout=(0, 0, 0),
    out_dnn_activation='tanh',
    temperature=0.05,
    dnn_use_bn=False,
    seed=1024,
    sampler_config=None,):
    
    features_columns = user_feature_columns + item_feature_columns
    # 构建 embedding_dict
    embedding_dict = build_embedding_dict(features_columns)

    # user 特征 处理
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns, embedding_dict, cate_map)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # item 特征 处理
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns, embedding_dict, cate_map)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)


    # user tower
    user_dnn_out = DNN(hidden_units=user_dnn_hidden_units, activation=out_dnn_activation, dropout_rates=user_dnn_dropout,
                       use_bn=dnn_use_bn, seed=seed, name='user_embed')(user_dnn_input)
    user_dnn_out = tf.nn.l2_normalize(user_dnn_out, axis=-1)


    # item tower
    item_dnn_out = DNN(hidden_units=item_dnn_hidden_units, activation=out_dnn_activation, dropout_rates=item_dnn_dropout,
                       use_bn=dnn_use_bn, seed=seed, name='item_embed')(item_dnn_input)
    item_dnn_out = tf.nn.l2_normalize(item_dnn_out, axis=-1)


    output = InBatchSoftmaxLayer(sampler_config=sampler_config._asdict(), temperature=temperature, name="dssm_out")(
            [user_dnn_out, item_dnn_out, item_features[sampler_config.item_name]])

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)


    return model