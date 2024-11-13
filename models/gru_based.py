from classification_models.keras import Classifiers
from tensorflow import keras
from tensorflow.keras.layers import GRU

SIZE = 299
DIM = (SIZE, SIZE, 3)
BATCH_SIZE = 150

def get_model():
    net, preprocess_input = Classifiers.get('xception')
    n_classes = 1

    base_model = net(input_shape=DIM, weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Reshape((1, -1))(x)
    x = GRU(64)(x)
    output = keras.layers.Dense(n_classes, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    
    return model


