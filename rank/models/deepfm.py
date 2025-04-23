import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from data.preprocessing import *
from tensorflow.keras.regularizers import l2

########################################################################
               #################定义模型##############
########################################################################
def DeepFM(linear_feature_columns, dnn_feature_columns, cate_map, fm_embed_dim=64, dnn_hidden_units=[512, 256, 128], dnn_activation='relu', seed=1024,):
    
    """Instantiates the DeepFM Network architecture.
    Args:
        linear_feature_columns: An iterable containing all the features used by linear part of the model.
        fm_group_columns: list, group_name of features that will be used to do feature interactions.
        dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        seed: integer ,to use as random seed.
        dnn_activation: Activation function to use in DNN
    return: A Keras model instance.
    """
    
    feature_columns = linear_feature_columns + dnn_feature_columns
    features = build_input_features(feature_columns)  
    inputs_list = list(features.values())


    # 构建 linear embedding_dict
    linear_embedding_dict = build_embedding_dict(linear_feature_columns, dim=1)
    linear_sparse_embedding_list, linear_dense_value_list = input_from_feature_columns(features, linear_feature_columns, linear_embedding_dict, cate_map)
    # linear part
    linear_logit = get_linear_logit(linear_sparse_embedding_list, linear_dense_value_list)

    # 构建 embedding_dict
    embedding_dict = build_embedding_dict(dnn_feature_columns, dim=fm_embed_dim)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, embedding_dict, cate_map)

    # 将所有sparse的k维embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，
    concat_sparse_kd_embed = Concatenate(axis=1, name="fm_concatenate")(sparse_embedding_list)  # ?, n, k
    # FM cross part
    fm_cross_logit = FMLayer()(concat_sparse_kd_embed)
    
    # DNN part
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    for i in range(len(dnn_hidden_units)):
        if i == len(dnn_hidden_units) - 1:
            dnn_out = Dense(units=dnn_hidden_units[i], activation=dnn_activation, kernel_regularizer=l2(1e-5), name='dnn_'+str(i))(dnn_input)
            break
        dnn_input = Dense(units=dnn_hidden_units[i], activation=dnn_activation, kernel_regularizer=l2(1e-5), name='dnn_'+str(i))(dnn_input)  
    dnn_logit = Dense(1, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(seed),name='dnn_logit')(dnn_out)

    final_logit = Add()([ linear_logit, fm_cross_logit, dnn_logit])

    output = tf.keras.layers.Activation("sigmoid", name="dfm_out")(final_logit)
    model = Model(inputs=inputs_list, outputs=output)

    return model





