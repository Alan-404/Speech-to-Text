from keras.models import Sequential
from keras.layers import Dense

def ffn(d_ff=2048, d_model=512, activation='relu'):
    return Sequential([
        Dense(units=d_ff, activation=activation),
        Dense(units=d_model)
    ])