import sys
import random
import numpy as np
import matplotlib.pyplot as plt

class neural_net:

    def __init__( self, X, t, fun, w0=None, layerdef=None,
                  name='neural-network' ):
        if X.shape[0] != t.shape[0]:
            raise ValueError( "First dimension of X must be same as t. "
                              "X.shape = {}, t.shape = {}."
                              .format(X.shape,t.shape) )
        self.name = name
        self.X = np.array(X)
        self.t = np.array(t)
        self.n_points = len(t)
        self.fun = fun
        if layerdef is None:
            # Default is no hidden layers
            self.layerdef = [len(X[0]),len(t[0])]
        else:
            l = layerdef
            l.append(len(t[0]))
            l.insert(0,len(X[0]))
            self.layerdef = l
        self.layers = len(self.layerdef)
        if w0 is None:
            # If weights not given, initialize 1 layer network with random
            # initial weights in [-1,1]. Add an extra bias weight at the end.
            np.random.seed(1)
            self.weights = []
            self.layer_err = []
            self.layer_del = []
            for k in range(self.layers - 1):
                self.weights.append( 2*np.random.random( ( self.layerdef[k+1],
                                                self.layerdef[k]   ) ) - 1 )
                self.layer_err.append(np.zeros(self.layerdef[k+1]))
                self.layer_del.append(np.zeros(self.layerdef[k+1]))
                self.layer = []
            for k in range(self.layers):
                self.layer.append(np.zeros(self.layerdef[k]))

    def feed_forward(self,n):
        self.layer[0] = self.X[n].T
        # Feed forward
        for k in range(self.layers-1):
            self.layer[k+1] = self.fun(np.dot(self.weights[k],self.layer[k]))

    def propagate_back(self,n):
        self.layer_err[-1] = self.t[n] - self.layer[-1]
        self.layer_del[-1] = ( self.layer_err[-1]
                               * self.fun(self.layer[-1],derivative=True) )
        self.weights[-1] += np.array( np.outer( self.layer[-2],
                                               self.layer_del[-1].T ) ).T
        for k in reversed(range(1,self.layers-1)):
            self.layer_err[k-1] = np.dot(self.weights[k].T,self.layer_del[k])
            self.layer_del[k-1] = ( self.layer_err[k-1]
                                    * self.fun(self.layer[k],derivative=True) )
            self.weights[k-1] += np.array( np.outer( self.layer[k-1],
                                                     self.layer_del[k-1].T ) ).T

    def train(self, iters=1000):
        progress_div = iters/100
        for k in range(self.layers):
            y = np.zeros(self.t.shape)
        for i in range(iters):
            for n in range(self.n_points):
                self.feed_forward(n)
                self.propagate_back(n)
                y[n] = self.layer[-1]
            if i % progress_div == 0:
                sys.stdout.write('Progress {} %\r'.format(int(i/iters*100)))
        return y

    def classify(self,X):
        y = []
        for n in range(len(X)):
            l = X[n].T
            # Feed forward
            for k in range(self.layers-1):
                l = self.fun(np.dot(self.weights[k], l))
            y.append(float(l))
        return y
        
def sigmoid(x, derivative=False):
    out = 1.0/(1.0 + np.exp(-x))
    if derivative:
        return out*(1 - out)
    return out

def tanh(x, derivative=False):
    if derivative:
        return 0.5*(1 - np.tanh(x)**2)
    return 0.5*np.tanh(x) + 0.5

def read_dataset(filename, skiplines=0):
    data  = []
    with open(filename) as f:
        for i in range(skiplines):
            f.readline()
        for l in f.readlines():
            data.append([float(x) for x in l.split()])
    return data

def look_at_data(X, skipcolumns=0):
    for i in range(skipcolumns, len(X.T)):
        for j in range(i+1, len(X.T)):
            if i == 0 or i % j != 0:
                plt.figure()
                plt.plot((X.T)[i], (X.T)[j], 'or')

def plot_solution(X, t, skipcolumns=0):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i in range(skipcolumns, len(X.T)):
        for j in range(i+1, len(X.T)):
            if i == 0 or i % j != 0:
                plt.figure()
                x = [ X[m][i] for m in range(len(X))]
                y = [ X[m][j] for m in range(len(X))]
                plt.scatter(x, y, c=t, s=50)
                plt.gray()
                plt.xlabel(i)
                plt.ylabel(j)

def main(argv):

    # X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) 
    # t = np.array([[0],[0],[1],[1]])
    # # t = np.array([[0],[1],[1],[0]])
    
    # nn = neural_net(X, t, sigmoid, layerdef=[4])
    # nn_out = nn.train(10000)
    # print(nn.classify(X))

    data = read_dataset('./old_faithful_classified.txt',1)
    # data = read_dataset('./dataset_3class_4feat.txt')
    data = data[0:50]
    training_frac = 0.8
    
    random.shuffle(data)
    X = np.array([row[1:] for row in data])
    t = np.array([[row[0]] for row in data])
    t = np.array([ z - m for m in [min(t)] for z in t])
    t = np.array([ z/s for s in [max(t)] for z in t])

    train_pts = int(training_frac*len(X))
    X_train = X[ 0 : train_pts ]
    X_test = X[ train_pts + 1 : -1 ]
    t_train = t[ 0 : train_pts ]
    t_test_truth = t[ train_pts + 1 : -1]

    nn = neural_net(X_train, t_train, tanh, layerdef=None)
    print(nn.layerdef)
    nn_out = nn.train(100000)
    t_test = nn.classify(X_test)
    err = np.subtract(t_test_truth, t_test)
    rms_err = np.sqrt(np.mean(err**2))
    print("RMS error {}".format(rms_err))

    # look_at_data(X)
    plot_solution(X_test, t_test)
    plt.show()

if __name__=="__main__":
    main(sys.argv[1:])

