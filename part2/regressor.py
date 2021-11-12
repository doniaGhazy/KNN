import numpy  as np
import matplotlib.pyplot as plt 
from numpy.linalg import inv



def poly_fit( Xtr, Ytr, M):
    Xtrain = Xtr
    for i in range(2,M+1):
        X_i = np.power(Xtr, i)
        Xtrain = np.append(Xtrain , X_i , axis = 1)
    X  = np.append(Xtrain, np.ones((Xtrain.shape[0], 1)) , axis = 1)    
    Xt = np.transpose(X)           
    Winv = inv(np.matmul(Xt , X))
    W = np.matmul(np.matmul(Winv,Xt),Ytr) 
    return W 

def predict( Xtr, W, M):   
    Xtrain = Xtr
    for i in range(2,M+1):
        X_i = np.power(Xtr, i)
        Xtrain = np.append(Xtrain , X_i , axis = 1)
    Xtrain  = np.append(Xtrain, np.ones((Xtrain.shape[0], 1)) , axis = 1) 
    y_predicted = np.matmul(np.transpose(W), np.transpose(Xtrain))
    return y_predicted

def error_rms( y , y_pred):
    N = len(y)
    err = y_pred - y
    err_square = np.power(err, 2)
    err_sum  = np.sum(err_square, axis  = 0)
    err_sum  = 0.5 * err_sum
    err_rms  = np.sqrt((2*err_sum) / N )
    return (err_rms)


def plot( X, Y, W, fig_num=0):
    plt.figure(fig_num)
    plt.figure(figsize=(6*3.13,4*3.13))
    plt.title("LS Model")
    print("x  " , X[:, 0])
    print("Y  " , Y[:, 0])
    plt.plot(X[:, 0], Y[:, 0], 'ro')
    x= np.linspace(30, 80, 100)
    y= W[0]
    for i in range (1, len(W)):
        y = y + (W[i] *x**(i))
    plt.plot(x,y,'co')
    plt.show()
