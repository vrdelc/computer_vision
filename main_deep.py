from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
from dataset_deep import Dataset

folder_dataset = 'C:/dataset/'

folder_dataset = 'crawler/'

IMG_SIZE = 700

#GENERATE DATASET
dataset = Dataset(img_size=IMG_SIZE)
dataset.show_statics()
dataset.prepare_data()
dataset.generate_model()

"""
# Test on Test Set
TEST_DIR = './test'
def load_test_data():
    test_data = []
    for img in os.listdir(TEST_DIR):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            test_data.append([np.array(img), label])
    shuffle(test_data)
    return test_data


test_data = load_test_data()
plt.imshow(test_data[10][0], cmap = 'gist_gray')

testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_data])

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print(acc * 100)

"""
