from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

DADADIR = f'{os.getcwd()}\\Data'

# Import data
with open(f'{DADADIR}\\X.pkl', 'rb') as file:
    X = pickle.load(file)
with open(f'{DADADIR}\\y.pkl', 'rb') as file:
    y = pickle.load(file)

# To np array 
y = np.array(y)

# Scale features
X = X /255.0

# Split data into training, validating and testing set
trainX, testX, trainy, testy = train_test_split(X,y, test_size=0.2, random_state=555)
trainX, valX, trainy, valy = train_test_split(trainX,trainy, test_size=0.15, random_state=296)


# # Build callbacks
early_stopping = EarlyStopping(    # Training kill switch
    monitor='val_loss',
    min_delta=0.005,
    patience=10,
    restore_best_weights=False
)

rlronp = ReduceLROnPlateau(    # Dynamic learning rate
    monitor='val_loss',
    factor=0.1,
    patience=3
)

# Define model name
TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
NAME = f'{TIME}_{X.shape[1:]}_Adam'

log_dir = f'Logs\\{NAME}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Build model structure and compile
model = Sequential([
    # Input Layer
    layers.Conv2D(130, kernel_size=(3,3) ,activation='relu', input_shape=X.shape[1:], padding='same'),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 1
    layers.Conv2D(105, kernel_size=(3,3),activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 2
    layers.Conv2D(145, kernel_size=(3,3),activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 3
    layers.Conv2D(115, kernel_size=(2,2), activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 4
    layers.Conv2D(165, kernel_size=(2,2), activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 5
    layers.Conv2D(75, kernel_size=(2,2), activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Hidden Layer 6
    layers.Conv2D(55, kernel_size=(2,2), activation='relu', padding='same'),
    # layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),

    # Output Layer
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(lr=0.001), 
    metrics=['accuracy'])

print(model.summary())

history = model.fit(trainX, trainy, validation_data=(valX,valy), batch_size=10, epochs=100,callbacks=[early_stopping, tensorboard_callback, rlronp])
pd.DataFrame(history.history).to_csv(f'{os.getcwd()}\\Data\\trainingHistory.csv',index=False)


# Save model
model.save(f'{os.getcwd()}\\Models\\{NAME}.h5')

# with open(f'{os.getcwd()}\\Models\\{NAME}.pkl', 'wb') as file:
#     pickle.dump(model, file)