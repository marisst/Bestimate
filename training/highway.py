# from https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535

import keras.backend as K
import keras.initializers
from keras.layers import Dense, Activation, Multiply, Add, Lambda
 
def highway_layers(value, n_layers, kernel_initializer, activation="tanh", kernel_regularizer=None, gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(value)
        transformed = Activation(activation)(value)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value