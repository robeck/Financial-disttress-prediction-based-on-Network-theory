'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现对比较模型的撰写和调用

'''
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit #导入logit模型
from sklearn.linear_model import LogisticRegression #导入logistic模型
from sklearn.naive_bayes import GaussianNB # 导入GaussianNB模型
from sklearn.ensemble import GradientBoostingClassifier # GBDT 梯度下降决策树
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline # 引入pipeline工具
from sklearn.metrics import accuracy_score,RocCurveDisplay # 使用sklearn计算准确率
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler # 标准化模型
from sklearn.model_selection import train_test_split # 数据划分
from sklearn.model_selection import permutation_test_score # 分析交叉验证排序测试
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

std = StandardScaler()
maxmin_scalar = MinMaxScaler()

"将数据截取，标准化的工作提取出来，单独来做"
def data_preprocess(traindata,preddata,std_tag):
    dataX,datay = np.array(traindata.iloc[:,:-1]),np.array(traindata.iloc[:,-1:]).ravel()
    predX,predy = np.array(preddata.iloc[:,:-1]),np.array(preddata.iloc[:,-1:]).ravel()

    "是否需要标准化？"
    if std_tag==True:
        dataX = std.fit_transform(dataX)
        predX = std.fit_transform(predX)
    elif std_tag == 'Max':
        dataX = maxmin_scalar.fit_transform(dataX)
        predX = maxmin_scalar.fit_transform(predX)
    else:
        pass

    return dataX,datay,predX,predy

'''logit模型
模型数据进行标准化！
'''
def logit_(datalist):
    '''
    :param preddata: 预测数据集，这里唯一用来存储数
    :param datalist: 数据要进行标准化！
    :return:
    '''
    # 数据处理部分
    dataX,datay,predX,predy = datalist

    "模型生成"
    model = Logit(endog=datay,exog=dataX,missing='drop')
    res = model.fit()
    print(res.summary())
    params = res.params  # 模型拟合后的参数，用于进一步的预测，绘制等工作
    "预测部分"
    prediction = model.predict(params=params,exog=predX,linear=False)
    pred_ST = [0 if x<0.5 else 1 for x in prediction] # 概率决策

    plt.plot(predy,color='red',label='True value')
    plt.plot(pred_ST,color='blue',label = 'Predicted Value')
    plt.show()

    acc = accuracy_score(predy,pred_ST)
    print(f'预测的准确度为：{acc}')

    return model


'''logistic模型
'''
def logistic_(datalist):
    '''
    :param datalist:
    :return: 数据不必精选标准化！
    '''
    # 数据处理部分
    dataX,datay,predX,predy = datalist
    print(f'The X shape is {dataX.shape}, Y shape is {datay.shape}')

    trainX,testX,trainy,testy = train_test_split(dataX,datay,train_size=0.7)
    print(f'The test x shape is {testX.shape},y shape is {testy.shape}')

    logist = LogisticRegression(penalty='none', tol=0.01, solver="saga")
    logist.fit(trainX,trainy)

    "2）Preddata数据集的结果验证"
    prediction = logist.predict(predX)
    prediction_prob = logist.predict_proba(predX)[:, 1]  # 预测值为1的那一列的概率！
    # 得分情况
    accuracy = accuracy_score(predy,prediction)
    precision = precision_score(predy, prediction, average='binary')
    recall = recall_score(predy, prediction, average='binary')
    f1_scores = f1_score(predy,prediction,average='binary')
    auc_scores = roc_auc_score(predy,prediction_prob)
    print(f'logistic模型的accuracy是{accuracy}')
    print(f'logistic模型的f1是{f1_scores}')
    print(f'logistic模型的auc是{auc_scores}')
    print(f'logistic模型的precision是{precision}')
    print(f'logistic模型的recall是{recall}')
    # parms = logist.get_params()
    # print(f'估计参数parameters：{parms}')


    return logist

'''简单SVC模型运用

'''
def svc_(datalist):
    '''
    :param datalist: 数据要进行标准化！
    :return:
    '''
    # 读取的数据处理
    dataX,datay,predX,predy = datalist

    "建模拟合（没有调参）"
    clf = SVC(kernel='linear',gamma='auto',probability=True)
    clf.fit(dataX,datay)

    "预测+绘制"
    prediction = clf.predict(predX) # 预测结果
    prediction_prob = clf.predict_proba(predX)[:, 1]  # 预测值为1的那一列的概率！
    "结果输出"
    accuracy = accuracy_score(predy,prediction)
    precision = precision_score(predy, prediction, average='binary')
    recall = recall_score(predy, prediction, average='binary')
    f1_scores = f1_score(predy,prediction,average='binary')
    auc_scores = roc_auc_score(predy,prediction_prob)
    print(f'SVC模型的accuracy是{accuracy}')
    print(f'SVC模型的f1是{f1_scores}')
    print(f'SVC模型的auc是{auc_scores}')
    print(f'SVC模型的precision是{precision}')
    print(f'SVC模型的recall是{recall}')

    "绘制"
    # svc_disp = RocCurveDisplay.from_estimator(clf,predX,predy)
    # plt.show()

    "测试一些交叉验证的permutation重要性"
    # cv = StratifiedKFold(2,shuffle=True,random_state=0)
    # score_og,prem_score_og,pvalue_og = permutation_test_score(
    #     clf,train
    # )

    return clf

''':Gaussan_NB模型
'''
def Navbay_(datalist):
    '''
    :param datalist: 数据不必进行标准化！
    :return:
    '''
    # 数据处理部分
    dataX,datay,predX,predy = datalist

    "模型"
    clf = GaussianNB()
    clf.fit(dataX,datay)
    # print(f'模型的参数为：{clf.get_params()}')

    "预测"
    prediction = clf.predict(predX)
    prediction_prob = clf.predict_proba(predX)[:, 1]  # 预测值为1的那一列的概率！
    "结果输出"
    accuracy = accuracy_score(predy,prediction)
    precision = precision_score(predy, prediction, average='binary')
    recall = recall_score(predy, prediction, average='binary')
    f1_scores = f1_score(predy,prediction,average='binary')
    auc_scores = roc_auc_score(predy,prediction_prob)
    print(f'Gaussan_NB模型的accuracy是{accuracy}')
    print(f'Gaussan_NB模型的f1是{f1_scores}')
    print(f'Gaussan_NB模型的auc是{auc_scores}')
    print(f'Gaussan_NB模型的precision是{precision}')
    print(f'Gaussan_NB模型的recall是{recall}')

    "绘制"
    # plt.plot(predy,color='red',label='test')
    # plt.plot(prediction,color='blue',label='prediction')
    # plt.show()
    #
    # ax = plt.gca()
    # NB_disp = RocCurveDisplay.from_estimator(clf, predX, predy, ax=ax, alpha=0.7)
    # plt.show()

    return clf

def gbdt_(datalist):
    # 数据处理部分
    dataX,datay,predX,predy = datalist
    # 模型
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(dataX,datay)
    # print(f'模型的参数为：{clf.get_params()}')

    "预测"
    prediction = clf.predict(predX)
    prediction_prob = clf.predict_proba(predX)[:, 1]  # 预测值为1的那一列的概率！
    "结果输出"
    accuracy = accuracy_score(predy,prediction)
    precision = precision_score(predy, prediction, average='binary')
    recall = recall_score(predy, prediction, average='binary')
    f1_scores = f1_score(predy,prediction,average='binary')
    auc_scores = roc_auc_score(predy,prediction_prob)
    print(f'GBDT模型的accuracy是{accuracy}')
    print(f'GBDT模型的f1是{f1_scores}')
    print(f'GBDT模型的auc是{auc_scores}')
    print(f'GBDT模型的precision是{precision}')
    print(f'GBDT模型的recall是{recall}')


    return clf

##########################################################################
"一个绘制calibration curve需要的方法，引用自sklearn"
class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba
##########################################################################


if __name__ == '__main__':
    pass