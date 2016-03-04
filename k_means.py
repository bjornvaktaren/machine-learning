import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from numpy import linalg
import numpy.random

class kmeans:

    def __init__(self, data, n_classes, mean_guess, name='k-means'):
        self.name = name
        self.data = np.array(data)
        self.n_classes = n_classes
        self.mean = np.array(mean_guess)
        self.classification = np.zeros((len(self.data), self.n_classes))

    def classify(self, iterations=1):
        for _ in range(iterations):
            for n in range(len(self.data)):
                d = np.linalg.norm(np.subtract(self.data[n], self.mean), axis=1)
                minval = min(d)
                for j in range(self.n_classes):
                    if d[j] == minval:
                        self.classification[n][j] = 1
            self.mean = ( np.dot( np.array(self.classification).T,
                                  np.array(self.data) ).T
                          / np.sum(self.classification, axis=0) ).T


def read_dataset(filename, skiplines=0):
    data  = []
    with open(filename) as f:
        for i in range(skiplines):
            f.readline()
        for l in f.readlines():
            data.append([float(x) for x in l.split()])
    return data

def look_at_data(X, skipcolumns=0):
    for i in range(skipcolumns,len(X.T)):
        for j in range(i,len(X.T)):
            if i % j != 0:
                plt.figure()
                plt.plot((X.T)[i], (X.T)[j], 'or')

def plot_solution(X, r, skipcolumns=0):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i in range(skipcolumns,len(X.T)):
        for j in range(i,len(X.T)):
            if i % j != 0:
                print('{} {}'.format(i,j))
                plt.figure()
                for n in range(len(r[0])):
                    x = [ X[m][i] for m in range(len(X)) if r[m][n]==1 ]
                    y = [ X[m][j] for m in range(len(X)) if r[m][n]==1 ]
                    plt.plot(x,y,'o',color=colors[n])

def main(argv):
    # data = read_dataset('./old_faithful_classified.txt')
    data = read_dataset('./dataset_3class_4feat.txt')
    # n_categories = 2
    # m0 = np.array([[-0.5,-0.5,-0.5,-0.5],[-0.0,-0.0,-0.0,0.0]])
    n_categories = 3
    m0 = np.array( [ [-0.5,-0.5,-0.5,-0.5],
                     [-0.0,-0.0,-0.0,0.0],
                     [0.5, 0.5, 0.5, 0.5] ] )
    iters = 10

    X = np.array([row[1:] for row in data])

    k = kmeans(X, n_categories, m0)
    k.classify(10)
    
    # look_at_data(X, 1)
    plot_solution(X, k.classification, 1)
    plt.show()

if __name__=="__main__":
    main(sys.argv[1:])
