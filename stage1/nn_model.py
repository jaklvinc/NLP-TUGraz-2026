from __future__ import annotations

import keras

def build_one_layer_nn(X,
                       y,
                       layer_size=64,
                       dropout=0.3):
    input_dim = X.shape[1]
    output_dim = y.shape[1] if len(y.shape) > 1 else 1

    if output_dim > 1:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    else:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'

    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(layer_size, 'relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(output_dim, activation=activation)
    ])

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    return model

class DeterministicNN(keras.wrappers.SKLearnClassifier):
    def __init__(self, model, seed=42, **kwargs):
        super().__init__(model=model, **kwargs)
        self.seed = seed

    def fit(self, X, y, **kwargs):
        keras.utils.set_random_seed(self.seed)
        return super().fit(X, y, **kwargs)
