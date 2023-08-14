'''
@Author: haozhi chen
@Date: 2022-08
@Target: 对于外部进入的数据，重新进行一下处理，使其符合模型运行的要求

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # 标准化实例

"Train,Test数据的处理"
def data_process1(dataframe):
    dataX,datay = np.array(dataframe.iloc[:,:-1]),np.array(dataframe.iloc[:,-1:])
    trainX,testX,trainy,testy = train_test_split(dataX,datay,test_size=0.3,random_state=0)

    return trainX,trainy.ravel(),testX,testy.ravel()


"Prediction数据的处理"
def data_process2(dataframe):
    dataX,data_y = np.array(dataframe.iloc[:,:-1]),np.array(dataframe.iloc[:,-1:])

    return dataX,data_y.ravel()


if __name__ == '__main__':
    pass