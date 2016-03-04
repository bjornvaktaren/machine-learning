import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from numpy import linalg
import numpy.random

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

def read_dataset(filename, skiplines=0):
    data  = []
    with open(filename) as f:
        for i in range(skiplines):
            f.readline()
        for l in f.readlines():
            data.append([float(x) for x in l.split()])
    return data

def look_at_data(X, skipcolumns=0):
    n = 0
    for i in range(skipcolumns,len(X.T)):
        for j in range(i,len(X.T)):
            if i % j != 0:
                plt.figure()
                plt.plot((X.T)[i], (X.T)[j], 'o', color=colors[n])
                plt.xlabel(i)
                plt.ylabel(j)
                n += 1

def plot_solution(X, r, skipcolumns=0):
    for i in range(skipcolumns,len(X.T)):
        for j in range(i,len(X.T)):
            if i % j != 0:
                print('{} {}'.format(i,j))
                plt.figure()
                for n in range(len(r[0])):
                    x = [ X[m][i] for m in range(len(X)) if r[m][n]==1 ]
                    y = [ X[m][j] for m in range(len(X)) if r[m][n]==1 ]
                    plt.plot(x, y, 'o', color=colors[n])

def main(argv):
    # data = read_dataset('./old_faithful_classified.txt')
    data = read_dataset('./dataset_3class_4feat.txt')

    X = np.array([row[1:] for row in data])
    
    look_at_data(X, 1)
    # plot_solution(X, r, 1)
    plt.show()

if __name__=="__main__":
    main(sys.argv[1:])
