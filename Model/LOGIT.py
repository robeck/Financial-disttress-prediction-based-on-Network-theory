'''
@Author: haozhic
@Date: 2022-07
@Target: 我们参考了一篇knowledge-based system的文章，使用logit模型来进行金融风险预警（公司是否被ST）。我们想进行一下复刻

我们以期希望，能够通过这个模型，对能源公司是否会被ST，是否会出现风险进行一个有效的预测
问题：
1）样本数量是否足够？能否给到足够的数据集

'''
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit #导入logit模型
from sklearn.metrics import accuracy_score # 使用sklearn计算准确率
import matplotlib.pyplot as plt

def read_data(flag1):
    data = []
    filename = '~/EnergyriskProject/Data/Energy_companies_trade_data.csv' # 能源公司数据
    df = pd.read_csv(filename)
    stocklists = list(set(df.Stkcd.tolist()))
    for i in range(0,flag1):
        stockid = stocklists[i] # 股票id
        stockdata = df[df.Stkcd==stockid].sort_values('Trddt') # 股票按照时间进行排序
        stockdata['Trddt'] = pd.to_datetime(stockdata.Trddt, format='%Y-%m-%d') # 股票的日期时间结构转换
        stockdata.set_index('Trddt',inplace=True) # 数据设置索引
        stockdata['prob'] = [0 if x<25 else 1 for x in stockdata.Clsprc] # 外生变量转换成概率（胜率）

        stockdata = stockdata[[ 'Opnprc', 'Hiprc', 'Loprc', 'Dnshrtrd', 'Dnvaltrd',
       'Dsmvosd', 'Dsmvtll','prob']]
        
        logit_model(stockdata)
        
    return data



def logit_model(data):
    # 数据描述
    print(data.describe())
    data['intercept'] = 1.0 # 手动添加常量数据
    col = ['Opnprc', 'Hiprc', 'Loprc','intercept'] # exog变量

    # 初始设置一下 训练，预测 数据集
    train_length = int(np.ceil((len(data)*0.8)))
    pred_length = int(np.ceil(len(data)))
    #
    traindata = data.iloc[0:train_length,:]
    predidata = data.iloc[train_length:,:]

    ''' 'logit 建模'
    logit模型是直接对胜率进行建模的！
    在logit模型的左边，是胜率的对数，右侧则是自变量的线性组合！
    
    因此：我们需要将数据进行转换，外生变量要转换成 “胜率”
    '''
    model = Logit(endog=traindata['prob'],exog=traindata[col],missing='drop')
    res = model.fit()
    print(res.summary())
    params = res.params  # 模型拟合后的参数，用于进一步的预测，绘制等工作

    prediction = model.predict(params=params,exog=predidata[col],linear=False)
    predidata['prediction'] = prediction
    predidata['pred_prob'] = [0 if x<0.5 else 1 for x in predidata.prediction.tolist()]
    print(predidata['prob'])
    print(predidata['pred_prob'])

    plt.plot(predidata['prob'],color='red')
    plt.plot(predidata['pred_prob'],color='blue')
    plt.show()

    acc = accuracy_score(predidata['prob'],predidata['pred_prob'])
    print(acc)

    return None



def main():
    flag1 = 1
    read_data(flag1)

    return None


if __name__ == '__main__':
    main()
