from pyrsistent import optional
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import pickle
import os

DADADIR = f'{os.getcwd()}\\Data'
# Import data dimensions
with open(f'{DADADIR}\\X.pkl', 'rb') as file:
    X = pickle.load(file)
input_shape = X.shape[1:]

del X

# Build model structures
def architect_1(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Dense(hp.Int('input_layer', min_value=16, max_value=256, step=16),
        activation=hp.Choice('activation', values=['relu','tanh']),
        input_shape=input_shape)
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv1_layer', min_value=16, max_value=256, step=25),
        activation=hp.Choice('activation', values=['relu','tanh']),
        kernel_size=(3,3))
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=25, max_value=625, step=25),
        activation=hp.Choice('activation', values=['relu','tanh']),
        kernel_size=(3,3))
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Layer 4 to 5
    model.add(layers.Flatten())
    for i in range(hp.Int('n_layers', 1, 2)):
        model.add(
            layers.Dense(hp.Int(f'{i}_layer', min_value=16, max_value=145, step=16),
            activation=hp.Choice('activation', values=['relu','tanh']))
        )

    # Output Layer
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    # Compile model
    model.compile(
        optimizer=Adam(lr=hp.Choice('lr', values=[0.001,0.0001,0.00001])),
        loss='binary_crossentropy', metrics=['accuracy']
    )

    return model


def architect_2(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Conv2D(hp.Int('input_layer', min_value=16, max_value=64, step=16),
        activation=hp.Choice('activation', values=['relu','tanh']),
        kernel_size=(3,3), input_shape=input_shape)
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)        
    )

    # Layer 2 to 3
    model.add(
        layers.Flatten()
    )
    for i in range(hp.Int('n_layers',1, 2)):
        model.add(
            layers.Dense(hp.Int(f'{i}_layer', min_value=25, max_value=100, step=25),
            activation=hp.Choice('activation', values=['relu', 'tanh']))
        )

    # Output Layer
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    # Compile model
    model.compile(
        optimizer=Adam(lr=hp.Choice('lr', values=[0.001,0.0001,0.00001])),
        loss='binary_crossentropy', metrics=['accuracy']
    )

    return model
    

def architect_3(hp):
    model = Sequential()
    activation_function = hp.Choice('activation', values=['relu','tanh'])
    lr = hp.Choice('lr', values=[0.001,0.0001,0.00001])

    # Layer 1
    model.add(
        layers.Dense(hp.Int('input_layer', min_value=5, max_value=25, step=5),
        activation=activation_function, input_shape=input_shape)
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv1_layer', min_value=10, max_value=75, step=5),
        kernel_size=(3,3), activation=activation_function)
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=10, max_value=75, step=5),
        kernel_size=(2,2), activation=activation_function)
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Output Layer
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    model.compile(
        optimizer=Adam(learning_rate=lr), 
        metrics=['accuracy'],
        loss='binary_crossentropy')

    return model


def architect_4(hp):
    model = Sequential()
    activation_function = hp.Choice('activation', values=['relu','tanh'])
    lr = hp.Choice('lr', values=[0.001,0.0001,0.00001])
    optional_layer = hp.Choice('optional_layer', values=[True, False])

    # Layer 1
    model.add(
        layers.Dense(hp.Int('input_layer', min_value=5, max_value=35, step=5),
        activation=activation_function, input_shape=input_shape)
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv1_layer', min_value=10, max_value=75, step=5),
        kernel_size=(3,3), activation=activation_function)
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=10, max_value=75, step=5),
        kernel_size=(2,2), activation=activation_function)
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )

    # Layer 4 (optional)
    if optional_layer:
        model.add(
            layers.Conv2D(hp.Int('conv3_layer', min_value=10, max_value=75, step=5),
            kernel_size=(3,3), activation=activation_function)
        )
        model.add(
            layers.MaxPool2D(pool_size=(2,2), strides=2)
        )

    # Output Layer
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    model.compile(
        optimizer=Adam(learning_rate=lr), 
        metrics=['accuracy'],
        loss='binary_crossentropy')

    return model


def architect_5(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Dense(hp.Int('input_layer', min_value=20, max_value=40, step=5),
        activation='relu', input_shape=input_shape)
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv1_layer', min_value=10, max_value=60, step=5),
        kernel_size=(3,3), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=10, max_value=60, step=5),
        kernel_size=(2,2), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 4
    model.add(
        layers.Conv2D(hp.Int('conv3_layer', min_value=10, max_value=60, step=5),
        kernel_size=(2,2), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Output Layer
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        metrics=['accuracy'],
        loss='binary_crossentropy')

    return model


def architect_6(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Dense(hp.Int('input_layer', min_value=35, max_value=100, step=5),
        activation='relu', input_shape=input_shape)
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv1_layer', min_value=45, max_value=115, step=5),
        kernel_size=(3,3), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=45, max_value=115, step=5),
        kernel_size=(2,2), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 4
    model.add(
        layers.Conv2D(hp.Int('conv3_layer', min_value=45, max_value=115, step=5),
        kernel_size=(2,2), activation='relu')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Output Layer
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        metrics=['accuracy'],
        loss='binary_crossentropy')

    return model



def architect_7(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Conv2D(hp.Int('convInput_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), input_shape=input_shape,
                      padding='valid')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), padding='valid')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv3_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), padding='valid')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
    layers.BatchNormalization()
    )

    # Layer 4
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(hp.Int('dense1_layer', min_value=50, max_value=150, step=5),
                     activation='relu')
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Output Layer
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model


def architect_8(hp):
    model = Sequential()

    # Layer 1
    model.add(
        layers.Conv2D(hp.Int('convInput_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), input_shape=input_shape,
                      padding='same')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 2
    model.add(
        layers.Conv2D(hp.Int('conv2_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), padding='same')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Layer 3
    model.add(
        layers.Conv2D(hp.Int('conv3_layer', min_value=50, max_value=150, step=5),
                      activation='relu', kernel_size=(3,3), padding='same')
    )
    model.add(
        layers.MaxPool2D(pool_size=(2,2), strides=2)
    )
    model.add(
    layers.BatchNormalization()
    )

    # Layer 4
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(hp.Int('dense1_layer', min_value=50, max_value=150, step=5),
                     activation='relu')
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.BatchNormalization()
    )

    # Output Layer
    model.add(
        layers.Dense(1, activation='sigmoid')
    )

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model