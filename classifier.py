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
        print("{} training and {} test".format(len(X_train),len(self.X_test)))
        X = self.clf.fit(X_train,y_train)
        # Save the SVM
        joblib.dump((self.clf, self.types_classes, self.histogramClasses[1], self.X_test, self.y_test), "bof.pkl", compress=3)

    def score(self):
        predictions = self.clf.predict(self.X_test)
        confussion_matrix = [[0,0],[0,0]]
        for i in range(len(self.y_test)):
            if self.y_test[i] == 0: #gambling
                if predictions[i] == 0: #TP
                    confussion_matrix[0][0] = confussion_matrix[0][0] +1
                else: #FN
                    confussion_matrix[1][0] = confussion_matrix[1][0] +1
            else: #non-gambling
                if predictions[i] == 0: #FP
                    confussion_matrix[0][1] = confussion_matrix[0][1] +1
                else: #TN
                    confussion_matrix[1][1] = confussion_matrix[1][1] +1
        print("Confusion matrix gambling: {}".format(confussion_matrix))
        precision = confussion_matrix[0][0] / (confussion_matrix[0][0]+confussion_matrix[0][1])
        recall = confussion_matrix[0][0] / (confussion_matrix[0][0]+confussion_matrix[1][0])
        f1_score = (2*precision*recall) / (precision+recall)
        print("Precision {:2.4f}".format(precision))
        print("Recall {:2.4f}".format(recall))
        print("F1 Score {:2.4f}".format(f1_score))
