import matplotlib.pyplot as plt 
import numpy     as np
import pandas    as pd
from itertools import combinations
def readData(file_path):
    """ data is a pandas dataframe with 5 columns and n rows  """
    # Reads data from all the excel file sheets                               
    data  = pd.read_excel(file_path, sheet_name= 'Sheet2')
    return data

def getColumn(df, col_names):              
    col = [df[col] for col in col_names]
    col = pd.concat(col, axis=1)
    return np.array(col)   

# slices dataframe to train and validation sets
def crossfold(data, n,i):                # n-fold , round-i
    set_len = len(data) // n
    valid_set = data[i*set_len : i*set_len+set_len]
    train_set = [data[:i*set_len], data[i*set_len+set_len: len(data)]]
    train_set = pd.concat(train_set, ignore_index= False)
    return train_set, valid_set

def comb(arr):
    comb = []
    for i in range(1,len(arr)+1):
        subset = [list(k) for k in combinations(arr,i)] 
        comb.append(subset)
    return comb

def erms(y,y_predicted):
     print(y)
     print(y_predicted)
     err = sum ((y_predicted - y)**2)
     err = err * 0.5
     rms = np.sqrt(2*(err/len(y))) 
     return rms
# plots error rms v.s poly. order
def plotrms(trms, vrms, fig_num):
    x  = range(1,len(trms)+1)
    plt.figure(fig_num)
    plt.xlabel('M')
    plt.ylabel('Erms')
    plt.plot(x,trms,'-o')
    plt.plot(x,vrms, '-o')
    plt.savefig('erms.png')
    plt.show()
