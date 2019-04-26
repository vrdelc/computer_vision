import os
import glob

class Dataset:

    def __init__(self, folder='crawler/'):
        self.folder = folder
        #List directory
        categories = os.listdir(self.folder)
        self.full_names_img = []
        self.number_images_categories= []
        self.list_labels = []
        for category in categories:
            path = (self.folder+category+"/")
            names_img = [os.path.join(path, f) for f in os.listdir(path)]
            number_img = len(names_img)
            print("{} images found for {}".format(number_img,category))
            #Append info to arrays
            self.full_names_img = self.full_names_img + names_img
            self.number_images_categories = self.number_images_categories+ [number_img]
            self.list_labels = self.list_labels + [category for f in range(0,number_img)]
        print("Total of {} images in the dataset".format(sum(self.number_images_categories)))
