import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from numpy import linalg
import numpy.random

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(np.subtract(x,y))**2 / (2 * (sigma ** 2)))

def read_dataset(filename):
    data  = []
    with open(filename) as f:
        f.readline()
        for l in f.readlines():
            d = l.split()
            data.append([float(d[0]), float(d[1]), float(d[2])])
    return data

def svm_train(X, t, C=None, kernel=gaussian_kernel):
    constr = {'type': 'eq',
              'fun': lambda x: sum([x[i]*t[i] for i in range(len(t))]) }
    # Change upper bound from None to a constant if there is
    # classification overlap.
    bnds = [(0,C) for i in range(len(t))] 
    x0 = np.random.uniform(size=len(t))
    def F(x):
        if x is None:
            return None
        L = 0
        for i in range(len(t)):
            L += x[i]
            for j in range(len(t)):
                L += -0.5*x[i]*x[j]*t[i]*t[j]*kernel(X[i],X[j])
        return -L
    return minimize(F, x0, bounds=bnds, constraints=constr)

def svm_classify(a, b, t, x, X, kernel):
    y = 0
    for i in range(len(a)):
        y += a[i]*t[i]*kernel(x,X[i])
    return y + b

def main(argv):
    data_train = read_dataset('./old_faithful_train_mixed.txt')
    data_test = read_dataset('./old_faithful_test.txt')
    kernel = linear_kernel
    # C determines the regularization in case of mixed classification.
    # C i basically the inverse of the regular lambda. 1/C in (0,1].
    C = 30

    # Prepare the data
    t = [row[0] for row in data_train]
    X = [[row[1], row[2]] for row in data_train]
    tn, xn, yn = [[row[i] for row in data_test] for i in range(3)]
    t1, x1, y1 = [[row[i] for row in data_train if row[0]==1] for i in range(3)]
    t2, x2, y2 = [ [row[i] for row in data_train if row[0]==-1]
                   for i in range(3) ]
    Xn = [[row[1], row[2]] for row in data_test]

    sol = svm_train(X, t, C=C, kernel=kernel)
    print(sol.message)
    a = sol.x
    Ns = 0
    b = 0
    # Classify the points in the test set.
    for i in range(len(a)): 
        if a[i] > 1e-6:
            b += t[i]
            for j in range(len(a)):
                b -= a[j]*t[j]*kernel(X[i],X[j])
            Ns += 1
    b = b/Ns

    tnew = []
    x1_new = []
    x2_new = []
    y1_new = []
    y2_new = []
    for z in Xn:
        y = svm_classify(a, b, t, z, X, kernel)
        if y < 0:
            tnew.append(-1.0)
            x2_new.append(z[0])
            y2_new.append(z[1])
        if y > 0:
            tnew.append(1.0)
            x1_new.append(z[0])
            y1_new.append(z[1])

    print('a = {}, b = {}, Ns = {}, tnew = {}'.format(a,b,Ns,tnew))
    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'ob')
    plt.plot(x1_new, y1_new, 'om')
    plt.plot(x2_new, y2_new, 'oc')
    plt.show()

if __name__=="__main__":
    main(sys.argv[1:])
