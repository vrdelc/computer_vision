from crawler import Crawler
from dataset import Dataset

folder_dataset = 'C:/dataset/'

#GENERATE DATASET
generate_dataset = False
if generate_dataset:
    crawler = Crawler()
    crawler.folder=folder_dataset
    crawler.generate()

#LOAD DATASET
dataset = Dataset(folder_dataset)
