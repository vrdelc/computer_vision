import os
import numpy as np
from scipy.cluster.vq import *
from sklearn.externals import joblib

class ImageInfo:

    def __init__(self,category,file_name,f1):
        self.category = category
        self.file_name =file_name
        self.descriptor = f1
        self.histogram = []

class Descriptors:

    def __init__(self, folder, load_vocabulary):
        self.folder = folder
        #List directory
        files = os.listdir(folder)
        self.data = []
        self.descriptors = []
        num_images = len(files)
        cont = 1
        #For each file in directory
        for file in files:
            print("Reading  descriptors of image {} of {}".format(cont,num_images))
            cont = cont+1
            if "_" not in file:
                if load_vocabulary:
                    #Load the vocabulary
                    self.vocabulary = np.loadtxt(self.folder+'vocabulary.gz', delimiter=',')
            else:
                #Generate the image information
                category = file.split('_')[0]
                file_name = file.split('_')[1].split(".")[0]
                f1 = np.loadtxt(folder+file, delimiter=',')
                descriptor = f1[0][2:]
                for i in range(1, len(f1)):
                    #Quit x,y position in image and append
                    descriptor = np.vstack((descriptor, f1[i][2:]))
                self.data.append(ImageInfo(category,file_name,descriptor))
                #Add descriptors to the general list
                if len(self.descriptors)==0:
                    self.descriptors = descriptor
                else:
                    self.descriptors = np.vstack((self.descriptors, descriptor))

    def generate_vocabulary(self, words):
        self.vocabulary, self.variance = kmeans(self.descriptors, words, 1)
        np.savetxt(self.folder+'vocabulary.gz', self.vocabulary, delimiter=',')

    def generate_histograms(self):
        # Calculate the histogram of features
        self.im_features = np.zeros((len(self.data), len(self.vocabulary)), "float32")
        self.im_classes = []
        self.im_names = []
        for i in range(len(self.data)):
            words, distance = vq(self.data[i].descriptor,self.vocabulary)
            for w in words:
                self.im_features[i][w] += 1
            self.im_classes.append(self.data[i].category)
            self.im_names.append(self.data[i].file_name)
        np.savetxt(self.folder+'histogramData.gz', self.im_features, delimiter=',')
        joblib.dump((self.im_classes, self.im_names), self.folder+'histogramClasses.gz')
