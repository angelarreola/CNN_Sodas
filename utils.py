import os
import numpy as np
import matplotlib.pyplot as ptl
import cv2
import pickle

data_dir="D:\\CNNSodas\\data\\sodas"

categories = ['coca','pepsi','sevenup']

data = []

def make_data():
    for category in categories:
        path = os.path.join(data_dir, category) 
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224,224))
                image = np.array(image, dtype=np.float32)
                
                data.append([image, label])
            except Exception as e:
                pass
    print(len(data))

    pik = open('data.pickle', 'wb')
    pickle.dump(data, pik)
    pik.close()

make_data()
