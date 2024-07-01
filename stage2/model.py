import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Dropout, Dense, Activation, Flatten, RepeatVector, Permute, multiply, Lambda, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from pykalman import KalmanFilter

def create_attention_model_GRU(input_shape, units):
    inputs = Input(shape=input_shape)
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Dropout(0.2)(cnn_out)
   
    gru_out = GRU(units, return_sequences=True)(cnn_out)
   
    attention = Dense(1, activation='relu')(gru_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([gru_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    output = Dense(1)(sent_representation)
    model = Model(inputs=inputs, outputs=output)
    return model

def simple_model_LSTM(input_shape, units):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units)(inputs)
    output = Dense(1)(lstm_out)
    model = Model(inputs=inputs, outputs=output)
    return model

def simple_model_GRU(input_shape, units):
    inputs = Input(shape=input_shape)
    gru_out = GRU(units)(inputs)
    output = Dense(1)(gru_out)
    model = Model(inputs=inputs, outputs=output)
    return model

def simple_model_BiLSTM(input_shape, units):
    inputs = Input(shape=input_shape)
    bilstm_out = Bidirectional(LSTM(units))(inputs)
    output = Dense(1)(bilstm_out)
    model = Model(inputs=inputs, outputs=output)
    return model

def temporal_fusion_transformer(input_shape, num_heads=4, d_model=64, dff=256, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Encoding layer
    x = Conv1D(filters=d_model, kernel_size=1, activation='relu')(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Self-attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed forward network
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    # Global average pooling
    gap_output = tf.keras.layers.GlobalAveragePooling1D()(out2)
    
    # Final dense layer
    output = Dense(1)(gap_output)
    
    model = Model(inputs=inputs, outputs=output)
    return model

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means