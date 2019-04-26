#Import libraries
from icrawler.builtin import GoogleImageCrawler
import os
from pixabay import Image
import urllib.request

class Crawler:

    #Crawler by defect, if not change parameters
    def __init__(self):
        self.folder = 'crawler/'
        self.threads = [2,4,8] #[feeder,parser,downloader]
        self.min_size = [200, 200]
        self.images_search = 30 #Total images = 2crawlers*2categories*4searches*images_search=16*images_search
        self.categories = 2
        self.categories_name = ['gambling','non-gambling']
        self.categories_search = [['gambling','poker','casino','slot machine'], ['board game','car game','block game','outside game']]

    def generate(self):
        print("Google crawler")
        self.google(init_id=0)
        print("Pixabay crawler")
        self.pixabay(init_id=(self.images_search*4))
        print("Generation is over")

    #Google crawler, init_id to know in which number begin to create images
    def google(self, init_id=0):
        #Foreach category
        for i in range(0, self.categories):
            category_name = self.categories_name[i]
            #print("Category name: {}".format(category_name))
            #For each search word
            for j in range (0,len(self.categories_search[i])):
                search_word = self.categories_search[i][j]
                #print(" - {}".format(search_word))
                #Search
                google_crawler = GoogleImageCrawler(feeder_threads=self.threads[0], parser_threads=self.threads[1], downloader_threads=self.threads[2],storage={'root_dir': self.folder+category_name})
                google_crawler.crawl(keyword=search_word, max_num=self.images_search, min_size=(self.min_size[0], self.min_size[1]),file_idx_offset=(self.images_search*j)+init_id)

    #Pixabay crawler, init_id to know in which number begin to create images
    def pixabay(self, init_id=0):
        #Key of the user(vriegc00)
        api_key = '12304969-479982339d49683b58ae74908'

        #It is necessary to add a version to download images
        class AppURLopener(urllib.request.URLopener):
            version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
        urllib._urlopener = AppURLopener()

        #Foreach category
        for i in range(0, self.categories):
            category_name = self.categories_name[i]
            #print("Category name: {}".format(category_name))
            #For each search word
            for j in range (0,len(self.categories_search[i])):
                search_word = self.categories_search[i][j]
                #print(" - {}".format(search_word))
                #Search
                image = Image(api_key)
                ims = image.search(q=search_word, page=1, per_page=self.images_search)
                #Download image one by one
                for image in ims['hits']:
                    extension = os.path.splitext(image['largeImageURL'])[-1]
                    urllib._urlopener.retrieve(image['largeImageURL'], self.folder+category_name+"/"+str(init_id).zfill(6)+extension)
                    init_id = init_id +1
