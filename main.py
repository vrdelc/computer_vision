from crawler import Crawler
from dataset import Dataset
from descriptor import Descriptors
import os

folder_dataset = 'C:/dataset/'
folder_descriptors = 'C:/descriptors/'
#folder_dataset = 'crawler/'

#Create folders that are neccessary if not exists
if not(os.path.isdir('files/')):
    os.mkdir('files')

#GENERATE DATASET
generate_dataset = False
if generate_dataset:
    crawler = Crawler()
    crawler.folder=folder_dataset
    crawler.generate()

#LOAD DATASET
load_dataset = False
if load_dataset:
    dataset = Dataset(folder_dataset)
    #Generate generate descriptors
    dataset.generate_descriptors()

#LOAD DESCRIPTORS
descriptors = Descriptors(folder_descriptors)
print(descriptors.data)
