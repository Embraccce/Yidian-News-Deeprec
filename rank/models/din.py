import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from data.preprocessing import *

def DIN(linear_feature_columns, dnn_feature_columns, query_feature_columns, key_feature_columns, bias_feature_columns, cate_map, hist_mask_value="0",
        dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="sigmoid",
        att_weight_normalization=True, dnn_dropout=0, seed=1024):
    
    """Instantiates the Deep Interest Network architecture.
    Args:
        dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        history_feature_names: list,to indicate  sequence sparse field
        dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
        dnn_activation: Activation function to use in deep net
        att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
        att_activation: Activation function to use in attention net
        att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
        dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        seed: integer ,to use as random seed.
    return: A Keras model instance.
    """
    feature_columns = list(set(linear_feature_columns + dnn_feature_columns + query_feature_columns + key_feature_columns))
    features = build_input_features(feature_columns)  
    inputs_list = list(features.values())

    # 构建 linear embedding_dict
    linear_embedding_dict = build_embedding_dict(linear_feature_columns, dim=1)
    linear_sparse_embedding_list, linear_dense_value_list = input_from_feature_columns(features, linear_feature_columns, linear_embedding_dict, cate_map)
    # linear part
    linear_logit = get_linear_logit(linear_sparse_embedding_list, linear_dense_value_list)


    # 构建 embedding_dict
    embedding_dict = build_embedding_dict(feature_columns)

    query_emb_list, _ = input_from_feature_columns(features, query_feature_columns, embedding_dict, cate_map)
    keys_emb_list, _ = input_from_feature_columns(features, query_feature_columns, embedding_dict, cate_map)
    bias_emb,_ = input_from_feature_columns(features, bias_feature_columns, embedding_dict, cate_map)
    dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(features, dnn_feature_columns, embedding_dict, cate_map)
    
    keys_emb = concat_func(keys_emb_list)
    query_emb = concat_func(query_emb_list)
    keys_seq = features[key_feature_columns[0].name]

    hist_attn_emb = AttentionPoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,hist_mask_value=hist_mask_value)([query_emb, keys_emb, keys_seq, bias_emb])
    dnn_input = combined_dnn_input(dnn_sparse_embedding_list+[hist_attn_emb], dnn_dense_value_list)

    # DNN
    for i in range(len(dnn_hidden_units)):
        if i == len(dnn_hidden_units) - 1:
            dnn_out = Dense(units=dnn_hidden_units[i], use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(seed),
                            activation=dnn_activation, name='dnn_'+str(i))(dnn_input)
            break
        dnn_input = Dense(units=dnn_hidden_units[i], use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(seed),
                          activation=dnn_activation, name='dnn_'+str(i))(dnn_input)  
    dnn_logit = Dense(1, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(seed),name='dnn_logit')(dnn_out)

    final_logit = Add()([ linear_logit, dnn_logit])
    output = tf.keras.layers.Activation("sigmoid", name="din_out")(final_logit)
    model = Model(inputs=inputs_list, outputs=output)

    return model