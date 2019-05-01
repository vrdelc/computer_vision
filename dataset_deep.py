import os
import glob
from SIFT.SIFT1 import SIFT_
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np

class Dataset:

    def __init__(self, folder='crawler/', img_size=300):
        self.folder = folder
        self.img_size = img_size
        #List directory
        self.categories_names = os.listdir(self.folder)
        self.heights = []
        self.widths = []
        self.images = []
        self.categories = []
        for category in self.categories_names:
            path = (self.folder+category+"/")
            names_img = [os.path.join(path, f) for f in os.listdir(path)]
            number_img = len(names_img)
            print("{} images found for {}".format(number_img,category))
            for img in os.listdir(self.folder+category+"/"):
                path = os.path.join(self.folder+category+"/", img)
                data = np.array(Image.open(path))
                self.heights.append(data.shape[0])
                self.widths.append(data.shape[1])
                img = Image.open(path)
                img = img.convert('L')
                img = img.resize((img_size, img_size), Image.ANTIALIAS)
                self.images.append(np.array(img))
                self.categories.append(category)
        self.avg_height = sum(self.heights) / len(self.heights)
        self.avg_width = sum(self.widths) / len(self.widths)

    def show_statics(self):
        print("Average Height: " + str(self.avg_height))
        print("Max Height: " + str(max(self.heights)))
        print("Min Height: " + str(min(self.heights)))
        print("Average Width: " + str(self.avg_width))
        print("Max Width: " + str(max(self.widths)))
        print("Min Width: " + str(min(self.widths)))
        print('\n')

    def prepare_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.categories, random_state=0)
        img_training = len(self.X_train)
        print("{} training images".format(img_training))
        img_test = len(self.X_test)
        print("{} test images".format(img_test))

        y_train_info = []
        for y in self.y_train:
            data = []
            for category in self.categories_names:
                data.append(int(category == y))
            y_train_info.append(data)
        self.train_data = np.array([i for i in self.X_train]).reshape(-1, self.img_size, self.img_size, 1)
        self.train_labels = np.array([i for i in y_train_info])

        self.test_labels = []
        for y in self.y_test:
            if self.categories_names[0] == y:
                self.test_labels.append(0)
            else:
                self.test_labels.append(1)
        self.test_data = np.array([i for i in self.X_test]).reshape(-1, self.img_size, self.img_size, 1)


    def generate_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 1)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        #self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation = 'softmax'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

        self.model.fit(self.train_data, self.train_labels, batch_size = 50, epochs = 20, verbose = 1, shuffle=True)

    def evaluate(self):
        predict_classes = self.model.predict_classes(self.test_data)

        confussion_matrix = [[0,0],[0,0]]
        for i in range(len(self.test_labels)):
            if self.test_labels[0] == 0: #gambling
                if predict_classes[i] == 0 : #TP
                    confussion_matrix[0][0] = confussion_matrix[0][0] +1
                else: #FN
                    confussion_matrix[1][0] = confussion_matrix[1][0] +1
            else: #non-gambling
                if predict_classes[i] == 0: #FP
                    confussion_matrix[0][1] = confussion_matrix[0][1] +1
                else: #TN
                    confussion_matrix[1][1] = confussion_matrix[1][1] +1
        print("Confusion matrix gambling: {}".format(confussion_matrix))
        precision = confussion_matrix[0][0] / (confussion_matrix[0][0]+confussion_matrix[0][1])
        recall = confussion_matrix[0][0] / (confussion_matrix[0][0]+confussion_matrix[1][0])
        f1_score = (2*precision*recall) / (precision+recall)
        print("Precision {:2.4f}".format(precision))
        print("Recall {:2.4f}".format(recall))
        print("F1 Score {:2.4f}".format(f1_score))
