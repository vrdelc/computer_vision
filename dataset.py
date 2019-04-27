import os
import glob
from SIFT.SIFT1 import SIFT_
from PIL import Image
import numpy as np

class Dataset:

    def __init__(self, folder='crawler/'):
        self.folder = folder
        #List directory
        categories = os.listdir(self.folder)
        self.data = []
        full_names_img = []
        self.number_images_categories= []
        list_labels = []
        for category in categories:
            path = (self.folder+category+"/")
            names_img = [os.path.join(path, f) for f in os.listdir(path)]
            number_img = len(names_img)
            print("{} images found for {}".format(number_img,category))
            #Append info to arrays
            full_names_img = full_names_img + names_img
            self.number_images_categories = self.number_images_categories+ [number_img]
            list_labels = list_labels + [category for f in range(0,number_img)]
        print("Total of {} images in the dataset".format(sum(self.number_images_categories)))
        self.data = [list_labels, full_names_img]

    def generate_descriptors(self, folder):
        for i in range(0,sum(self.number_images_categories)):
            file_name = self.data[1][i]
            category = self.data[0][i]
            id_image = file_name.split('/')[-1].split(".")[0]
            #Read image
            img01 = Image.open(file_name)
            #Convert from RGB to grayScale
            img1 = img01.convert('L')
            try:
                #Create a SIFT class with the image
                s1 = SIFT_(np.array(img1))
                #Obtain the descriptors
                f1 = s1.get_features().get_descriptor()
                #Save in a compressed file
                np.savetxt(folder+category+'_'+id_image+'.gz', f1, delimiter=',')
            except:
                print("SIFT error in image: {}".format(file_name))
