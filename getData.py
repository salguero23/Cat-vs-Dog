from tqdm import tqdm
import numpy as np
import pickle 
import random
import cv2
import os

DATADIR = f'{os.getcwd()}\\PetImages'
CATEGORIES = ['Cat','Dog']
data = []


def create_data(IMG_SIZE=50, color=False, seed=267):
    for category in CATEGORIES:
        # Locate data directory and label dummy variable
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        # Iterate through each file in the path
        for img in tqdm(os.listdir(path)):
            try:
                if color: # Logic to keep 3 dimensions (i.e. color) or convert to black & white
                    img_array = cv2.imread(os.path.join(path,img))
                else:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                
                # Resize image to normalize data
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([img_array, class_num])
            except:
                pass

    # Shuffle data
    random.seed(seed)
    random.shuffle(data)

    # Pickle our data into features and label files
    X, y = [], []

    for features, label in data:
        X.append(features)
        y.append(label)

    # Convert features to np array and reshape
    if color: # Logic for reshaping color data or balck & white data
        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    else:
        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Save data
    with open(f'{os.getcwd()}\\Data\\X.pkl', 'wb') as file: # Feature data
        pickle.dump(X, file)
    with open(f'{os.getcwd()}\\Data\\y.pkl', 'wb') as file: # Label data
        pickle.dump(y, file)

# Run function
create_data(IMG_SIZE=115,color=False)