import numpy as np
import operator
from collections import Counter
import itertools
def CountFrequency(my_list): 
    # Creating an empty dictionary  
        merged = list(itertools.chain(*my_list))
        freq = {} 
        for item in merged: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1

        return freq

class nn(object):
    def _init_(self):
        pass
    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y


    # X is training data == X[foldTrain]
    # X_ testing data == y[foldTest]
    # y is training output
    def predict(self, X_ , nn, dist):
        num_test = X_.shape[0]
        #get the neighbours
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            if dist == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X_[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X_[i,:]), axis = 1))


            min_indices = distances.argsort()[:nn]
            min_labels = []
            for index in min_indices:
                min_labels.append(self.ytr[index])

            freq_dict = CountFrequency(min_labels)
            pred_lbl = max(freq_dict.items(), key=operator.itemgetter(1))[0]
            Ypred[i] = pred_lbl # predict the label of the nearest example

        return Ypred