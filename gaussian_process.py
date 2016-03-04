import sys
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st
from operator import add

def kernel(x, xp, theta):
    return sp.exp( -1.0*(x - xp)*(x - xp)/theta )

def gram_matrix(x_vals, kernel, *kernel_params):
    gram = []
    for xn in x_vals:
        for xm in x_vals:
            gram.append(kernel(xn, xm, *kernel_params))
    return sp.array(sp.reshape(gram, (len(x_vals), len(x_vals))))

def gaussian_process_regression(x, y, new_x_vals, beta, kernel, *kernel_params):
    # Compute the Gram matrix
    k = gram_matrix(x, kernel, *kernel_params)
    # For each new point N+1 calc p(t_(N+1)|T) = Gauss(t_(N+1)|m,c) where
    # m = , c = , and l such that l_n = kernel(x_(N+1), x_n)
    p = []
    for new_x in new_x_vals:
        l_tmp = []
        for xn in x:
            l_tmp.append(kernel(new_x, xn, *kernel_params))
        # This l is actually already the transpose
        l_t = sp.array(l_tmp)
        a = sp.linalg.inv(k + 1.0/beta*sp.identity(len(x)))
        m = l_t.dot(a).dot(y)
        c = ( kernel(new_x, new_x, *kernel_params) + 1.0/beta
              - l_t.dot(a).dot(l_t.T) )
        p.append(st.norm(m,sp.sqrt(c)))
    return p

def main(argv):
    x = [9.8, 15.4, 7.9, 5.4, 0.7]
    y = [0.1, 2.1, 1.3, -1.7, -0.01]
    new_x = sp.linspace(0, 20, 200)
    
    p = gaussian_process_regression(x, y, new_x, 100.0e9, kernel, 20)
    pmean = sp.array([ pp.mean() for pp in p ])
    pstd2 = sp.array([ 2.0*pp.std() for pp in p ])
    pstd = sp.array([ pp.std() for pp in p ])
    
    fig, ax = plt.subplots(1,1)
    ax.fill_between( new_x, pmean - pstd2, pmean + pstd2,
                     facecolor='gray', alpha=0.5, color='gray' )
    ax.fill_between( new_x, pmean - pstd, pmean + pstd,
                     facecolor='gray', color='gray', alpha=0.5 )
    ax.plot( new_x, pmean, color='red' )
    ax.plot(x, y, 'o', color='black')


    plt.show()

if __name__=="__main__":
    main(sys.argv[1:])
