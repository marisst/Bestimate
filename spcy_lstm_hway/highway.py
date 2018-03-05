# from https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535

import keras.backend as K
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers
 
def highway_layers(value, n_layers, activation="tanh", gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim)(value)
        transformed = Activation(activation)(value)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value
