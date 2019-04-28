import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import csv
from sklearn.model_selection import train_test_split

class Classifier:

    def __init__(self,folder_histogram):
        self.histograms = np.loadtxt(folder_histogram+'histogramData.gz', delimiter=',')
        self.histogramClasses = joblib.load(folder_histogram+'histogramClasses.gz')
        self.classes = []
        self.types_classes = []
        for class_name in self.histogramClasses[0]:
            if class_name not in self.types_classes:
                self.types_classes.append(class_name)
            self.classes.append(self.types_classes.index(class_name))
        # Train the Linear SVM
        self.clf = LinearSVC()
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.histograms, self.classes, random_state=0)
        X = self.clf.fit(X_train,y_train)
        # Save the SVM
        joblib.dump((self.clf, self.types_classes, self.histogramClasses[1], self.X_test, self.y_test), "bof.pkl", compress=3)

    def score(self):
        score = self.clf.score(self.X_test,self.y_test)
        print("Mean accuracy to classifier: {:2.2f}%".format(score*100))
