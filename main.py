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
    crawler.generate(folder_descriptors)

#LOAD DATASET
generate_descriptors = False
if generate_descriptors:
    dataset = Dataset(folder_dataset)
    #Generate generate descriptors
    dataset.generate_descriptors()

#LOAD DESCRIPTORS AND VOCABULARY
generate_vocabulary = False
vocabulary_words = 10
descriptors = Descriptors(folder_descriptors, not(generate_vocabulary))
if generate_vocabulary:
    descriptors.generate_vocabulary(vocabulary_words)
print("Word in vocabulary: {}".format(len(descriptors.vocabulary)))

#GENERATE HISTOGRAMS
descriptors.generate_histograms()
