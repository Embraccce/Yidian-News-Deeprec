import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from data.preprocessing import *
from tensorflow.keras.regularizers import l2

def ESMM(linear_feature_columns, fm_cross_columns, dnn_feature_columns, cate_map, fm_embed_dim=16, num_tasks=2, tasks=["binary", "binary"], 
         tasks_name=["ctr","cvr"], num_experts=3, units_experts=256, task_dnn_units=(128, 64), seed=1024, dnn_activation='relu', l2_reg=1e-5,):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    Args:
        dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
        tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
        num_experts: integer, number of experts.
        units_experts: integer, the hidden units of each expert.
        tasks_name: list of str, the name of each tasks,
        task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
        dnn_activation: Activation function to use in DNN
    return: return a Keras model instance.
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))
        
    
    feature_columns = linear_feature_columns + fm_cross_columns + dnn_feature_columns
    features = build_input_features(feature_columns)  
    inputs_list = list(features.values())


    # 构建 linear embedding_dict
    linear_embedding_dict = build_embedding_dict(linear_feature_columns, dim=1)
    linear_sparse_embedding_list, linear_dense_value_list = input_from_feature_columns(features, linear_feature_columns, linear_embedding_dict, cate_map)
    # linear part
    linear_logit = get_linear_logit(linear_sparse_embedding_list, linear_dense_value_list)

    # 构建 FM embedding_dict
    fm_embedding_dict = build_embedding_dict(fm_cross_columns, dim=fm_embed_dim)
    fm_sparse_embedding_list, _ = input_from_feature_columns(features,  fm_cross_columns, fm_embedding_dict, cate_map)
    # 将所有sparse的k维embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，
    concat_sparse_kd_embed = Concatenate(axis=1, name="fm_concatenate")(fm_sparse_embedding_list)  # ?, n, k
    # FM cross part
    fm_cross_logit = FMLayer()(concat_sparse_kd_embed)
    
    # DNN part
    dnn_embedding_dict = build_embedding_dict(dnn_feature_columns)
    dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 dnn_embedding_dict, cate_map)
    dnn_input = combined_dnn_input(dnn_sparse_embedding_list, dnn_dense_value_list)

    # MMOELayer
    mmoe_layers = MMoELayer(units_experts=units_experts, num_tasks=num_tasks, num_experts=num_experts,
                            name='mmoe_layer')(dnn_input)

    # 分别处理不同 Task Tower
    task_outs = []
    for task_layer, task, task_name in zip(mmoe_layers, tasks, tasks_name):
        tower_ouput = DNN(hidden_units=task_dnn_units, activation=dnn_activation, l2_reg=l2_reg, seed=seed, name=f'tower_{task_name}')(task_layer)
        
        # batch_size * 1
        dnn_logit = Dense(units=1, activation=None, use_bias=False,
                                      kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                      kernel_regularizer=l2(l2=l2_reg),
                                      name=f'logit_{task_name}')(tower_ouput)
        
        final_logit = Add()([ linear_logit, fm_cross_logit, dnn_logit])
        
        output = PredictionLayer(task, name=task_name)(final_logit)

        task_outs.append(output)

    # esmm
    ctr_pred = task_outs[0]
    cvr_pred = task_outs[1]

    # CTCVR = CTR * CVR
    ctcvr_pred = Multiply(name="ctcvr")([ctr_pred, cvr_pred])
    task_outputs = [ctr_pred, ctcvr_pred]

    model = Model(inputs=inputs_list, outputs=task_outputs)

    return model