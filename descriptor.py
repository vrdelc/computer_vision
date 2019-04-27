import os
import numpy as np

class Descriptors:

    def __init__(self, folder):
        #List directory
        files = os.listdir(folder)
        self.data = []
        #For each file in directory
        for file in files:
            category = file.split('_')[0]
            file_name = file.split('_')[1].split(".")[0]
            f1 = np.loadtxt(folder+file, delimiter=',')
            self.data.append([category,file_name,f1])
