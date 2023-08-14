'''
@Author: haozhi chen
@Date: 2022-02
@Target: 主要工作就是提供一个金融指标的前期特征选择解决方案，模块化执行，可以扩充和缩减

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # RF 分类算法
from sklearn.feature_selection import RFE # RFE 方法
from Model.PSO_SVM.Config import args, kernel
import matplotlib.pyplot as plt

from feature_engine.selection import RecursiveFeatureElimination # 第三方的特征消除

"RandomForest的特征选择方法"
''':说明
1）：针对的是用于SVM模型所使用的数据，列均是序号
2）：如果要是用dataframe结构，需要重新考虑
'''
def randomforest_feature_selection(TrainX,Trainy,predX,finposition,features):
    selected_features = None
    # 提取Fin，NonFin特征
    FinTrainX, NonFinTrainX = TrainX[:, finposition:], TrainX[:, 0:finposition] #这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分
    # RandomForest 模型
    rfc = RandomForestClassifier(n_estimators=100, random_state=12)
    # 训练模型
    rfc.fit(FinTrainX, Trainy)
    # 获取特征重要性
    importances = rfc.feature_importances_ # 从小到大排序
    # 对特征重要性排序
    indices = np.argsort(importances)[::-1] # 从大到小排序
    # 选择重要性最高的特征,前10个
    selected_features = indices[:features] # 可调整测试
    # column排序
    selected_features.sort()
    # 选择后的金融特征指标
    NewFinTrainX = FinTrainX[:, selected_features]
    # 合并Fin NonFin
    TrainX = np.c_[NewFinTrainX, NonFinTrainX]

    # pred 根据前面的特征重要性重构数据
    FinPredX, NonFinPredX = predX[:, finposition:], predX[:, 0:finposition]  #这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分
    NewFinPredX = FinPredX[:, selected_features]
    predX = np.c_[NewFinPredX, NonFinPredX]  # 生成新的PredX数据

    print(f'The selected financial features in colunm list are {selected_features}')
    return TrainX,predX

"RFE递归特征选择方法"
''':说明
1）：针对的是用于SVM模型所使用的数据，列均是序号
2）：模型是基于SVC的基础，因此存在导入SVC参数的情况
2）：如果要是用dataframe结构，需要重新考虑
'''
def recursive_featres_elimination_svc(TrainX,Trainy,predX,SVCparmams,finposition,features):

    # 提取Fin，NonFin特征
    FinTrainX, NonFinTrainX = TrainX[:, finposition:], TrainX[:, 0:finposition] #这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分

    "1. 构建模型，筛选数据即，拟合"
    '''
    基本思路是：
    1）先使用基础的SVC来过滤有效的变量，参数
    2）使用优化过参数的模型，用SVC过滤的数据集进行操作
    '''
    classifer = SVC(kernel='linear', gamma=SVCparmams[0], C=SVCparmams[1], random_state=66,
                     probability=True)  # 非pipline模型

    # 模型RFE选择器，基于SVC
    selector = RFE(estimator=classifer, n_features_to_select=features,step=1) #控制选择的特征个数！可调整测试
    # 特征选择器训练
    selector.fit(FinTrainX, Trainy)  # 从金融指标 <-> target中 找到合适的指标！
    # 选取新的指标
    NewfintrainX = FinTrainX[:,selector.support_]  # Reduce X to the selected features 形成新的features
    # 合并Fin NonFin
    TrainX = np.c_[NewfintrainX, NonFinTrainX]  # 训练数据X

    # pred 根据前面的特征重要性重构数据
    FinPredX, NonFinPredX = predX[:, finposition:], predX[:, 0:finposition]  # 这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分
    NewFinPredX = FinPredX[:, selector.support_] # 选择最后特征剩下的列
    predX = np.c_[NewFinPredX, NonFinPredX]  # 生成新的PredX数据

    return TrainX,predX

"RFE递归特征选择方法"
''':说明
1）：针对的是用于RandomForest模型所使用的数据，列均是序号
2）：模型是基于randomforest的基础,需要导入rf模型
2）：输入输出的数据均为dataframe结构
'''
def recursive_featres_elimination_rf(TrainX,Trainy,predX,Model,finposition,features):
    # 提取Fin，NonFin特征:dataframe结构
    FinTrainX, NonFinTrainX = TrainX.iloc[:, finposition:], TrainX.iloc[:, 0:finposition] #这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分
    FinpredX, NonFinpredX = predX.iloc[:, finposition:], predX.iloc[:, 0:finposition]  # 这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分 预测数据部分
    print('拆分数据的shape')
    print(FinTrainX.shape)
    print(NonFinTrainX.shape)
    print(FinTrainX.head())
    print(NonFinTrainX.head())
    # 选择器
    rfe = RFE(Model,n_features_to_select=features,step=1) # 调整选择特征的个数
    # 拟合
    rfe.fit(FinTrainX,Trainy)
    # 输出选中的特征
    Finindicators = FinTrainX.columns[rfe.support_]
    print(f"selected featuress: {Finindicators}")
    # 构建新的选中的特征 金融特征
    Select_finTrainX = FinTrainX.iloc[:,rfe.support_]
    Select_finpredX = FinpredX.iloc[:, rfe.support_]
    # print('重构数据的shape')
    # print(Select_finTrainX.shape)
    # print(Select_finTrainX.head())
    # 合并
    NewTrainX = pd.concat([Select_finTrainX,NonFinTrainX],axis=1)
    NewpredX = pd.concat([Select_finpredX,NonFinpredX],axis=1)

    return Finindicators, NewTrainX, NewpredX

"RFE递归特征选择方法"
''':说明
1）：针对的是用于RandomForest模型所使用的数据，列均是序号
2）：模型是基于randomforest的基础,需要导入rf模型
2）：输入输出的数据均为dataframe结构
'''
def feature_engine_rfe(TrainX,Trainy,predX,Model,finposition,features,threshold):
    # 提取Fin，NonFin特征:dataframe结构
    FinTrainX, NonFinTrainX = TrainX.iloc[:, finposition:], TrainX.iloc[:, 0:finposition] #这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分
    FinpredX, NonFinpredX = predX.iloc[:, finposition:], predX.iloc[:, 0:finposition]  # 这个位置是要学会调整的，因为当特征变量多的时候，并非5这个位置划分 预测数据部分
    print('拆分数据的shape')
    print(FinTrainX.shape)
    print(NonFinTrainX.shape)
    print(FinTrainX.head())
    print(NonFinTrainX.head())
    # 选择器
    rfe = RecursiveFeatureElimination(
        variables = None,
        estimator = Model,
        scoring='accuracy',# the metric we want to evalute
        threshold=threshold,# the maximum performance drop allowed to remove a feature 0.0005
        cv=2
    )

    # 拟合
    rfe.fit(FinTrainX,Trainy)
    #
    print(f'performance of model trained using all features: {rfe.initial_model_performance_}')
    # 绘制
    rfe.feature_importances_.plot.bar(figsize=(20, 6))
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()
    # 丢弃的特征
    print(f'the droped features : {rfe.features_to_drop_}')
    # 输出选中的特征
    print(f'the total of features : {rfe.feature_names_in_}')
    FinIndicators = [features for features in rfe.feature_names_in_ if features not in rfe.features_to_drop_]
    print(f'the number of total of features : {len(FinIndicators)}')
    # 构建新的选中的特征 金融特征
    Select_finTrainX = rfe.transform(FinTrainX)
    Select_finpredX = rfe.transform(FinpredX)
    # print('重构数据的shape')
    # print(Select_finTrainX.shape)
    # print(Select_finTrainX.head())
    # 合并
    NewTrainX = pd.concat([Select_finTrainX,NonFinTrainX],axis=1)
    NewpredX = pd.concat([Select_finpredX,NonFinpredX],axis=1)

    return FinIndicators,NewTrainX, NewpredX

"提前预备一个唤起函数"
def selection_process(TrainX,Trainy,predX,finposition):
    return randomforest_feature_selection(TrainX,Trainy,predX,finposition)


if __name__ == '__main__':
    # selection_process()
    pass