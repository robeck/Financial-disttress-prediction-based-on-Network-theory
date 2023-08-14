''':从基础模型入手
@Author : haozhi chen
@Date : 2023-03-02
@Target : 实现一个基础的适用GridSearch进行优化的SVM模型

GridSearch SVM model
'''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler

def gridsearch_svc(date,traindata,preddata):
    sc = StandardScaler() # 初始化一个标准化模型
    TrainX,Trainy = traindata
    predX, predy = preddata
    # 标准化数据
    TrainX = sc.fit_transform(TrainX)
    predX = sc.fit_transform(predX)

    # 定义model
    svc = SVC(probability=True)
    # 定义参数网格
    parameters = {'kernel': ('linear', 'rbf'), 'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
    # 定义网格搜索对象
    clf = GridSearchCV(svc,parameters)
    # 拟合数据
    clf.fit(TrainX,Trainy)

    # 输出一下模型最优参数
    print(f'----------在日期为 {date} 的时期 grid search 得到的最优参数为：{clf.best_params_}')
    print(f'----------在日期为 {date} 的时期 grid search svc得到的最优得分为：{clf.best_score_}')

    # 测试一下预测
    prediction_pred = clf.predict(predX)
    pred_prov_pred = clf.predict_proba(predX)[:,1]
    # 相关得分
    scores = clf.score(predX,predy)
    print(f'---------在日期为 {date} 的时期 grid search svc得到的最优得分为：{scores}')


    return None