from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
from dataset_deep import Dataset

folder_dataset = 'C:/dataset/'

#folder_dataset = 'crawler/'

IMG_SIZE = 700

#GENERATE DATASET
dataset = Dataset(img_size=IMG_SIZE)
dataset.show_statics()
dataset.prepare_data()
dataset.generate_model()
dataset.evaluate()
