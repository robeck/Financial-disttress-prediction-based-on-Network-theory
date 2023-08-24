''':
@Author: haozhi chen
@Date: 2022-10
@Target: 实现使用PSO算法优化RandomForest的参数

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # 网格搜索的参数优化
from sklearn.model_selection import cross_val_score # 学习曲线寻优
from sklearn.metrics import roc_auc_score # 计算auc得分
from sklearn.metrics import RocCurveDisplay # 即将替代上面的plot_roc_curve
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score,roc_curve,auc
from sklearn.metrics import PrecisionRecallDisplay # 绘制precision-recall

from Results_Plot.KS_curve import plt_ks
from Model import Feature_selection

import matplotlib.pyplot as plt
import random
import MyError
from tqdm import tqdm
import shap

from Model.PSO_RandomForest.PSO_Config import args


"输入的数据进行预处理"
''':param
tags: train_test 划分数据集，划分比例7:3
    : normal 不划分数据集
'''
def input_data_process(data,tags):
    X,y = data.iloc[:,0:-1],data.iloc[:,-1]
    if tags == 'train_test':
        return train_test_split(X,y,random_state=0,test_size=0.3)
    elif tags == 'normal':
        return X,y
    else:
        raise MyError.Myexception('RandomForest数据处理遇到了问题')
    


"适应度函数"
def fittness_function(params,data):
    train_X,test_X,train_y,test_y = data
    clf = RandomForestClassifier(n_estimators=params[0],
                                 max_depth=params[1],
                                 min_samples_leaf=params[2],
                                 criterion='gini',
                                 random_state=0,
                                 n_jobs=-1)
    clf.fit(train_X,train_y)
    y_train_pred = clf.predict(train_X)
    y_test_pred =clf.predict(test_X)

    return confusion_matrix(train_y, y_train_pred)[0][1] + confusion_matrix(train_y, y_train_pred)[1][0], \
           confusion_matrix(test_y, y_test_pred)[0][1] + confusion_matrix(test_y, y_test_pred)[1][0]


"PSO不断优化RandomForest的参数"
def PSO_RF_Model(data):
    # 初始化参数
    iteration = 0  # 初始迭代标记
    '''
    参数1：代表粒子位置，其实也是适应函数输入的参数（n_estimators,max_depth,min_sample_leaf)
    n_estimators : 0-200之间
    max_depth: 0-30
    min_sample_leaf: 0-20
    '''
    particle_position_vector = np.array([np.array([random.randint(1,200),
                                                   random.randint(1,30),
                                                   random.randint(1,20)]) for _ in range(args.n_particles)]) # 初始化每一个粒子的位置
    # 参数2：粒子自身历史最优位置
    pbest_position = particle_position_vector
    # 参数3：粒子自身最优的适应函数值 初始化为 0
    pbest_fitness_value = np.array([float('inf') for _ in range(args.n_particles)])
    # 参数4，5：全局位置初始，全局适应函数初始化 0,0,0
    gbest_fitness_value = np.array([float('inf'),float('inf'),float('inf')])
    gbest_position = np.array([float('inf'),float('inf'),float('inf')])
    # 参数6：速度向量初始化
    velocity_vector = ([np.array([0,0,0]) for _ in range(args.n_particles)])

    '进行不断的迭代'
    while iteration < args.n_iterations:
        print(f'进行迭代中，轮数为：{iteration}')
        # plot(particle_position_vector) # 绘制初始化的粒子分布散点图
        '遍历100个粒子'
        for i in tqdm(range(args.n_particles)):
            fitness_res = fittness_function(particle_position_vector[i],data) #统计预测结果
            # print("error of priticle ",i,'is (training,test)',fitness_res,"At (n_estimators,max_depth,min_sample_leaf): ",
            #       particle_position_vector[i]) # 参数的输出显示

            """
            初始化的 自身历史最优 进行迭代替换
            （1）比较
            （2）用较好结果 替换 自身历史最优：这是一个自己比较的过程
            （3）粒子位置信息（参数）替换
            """
            if (pbest_fitness_value[i] > fitness_res[1]): # 因为初始的局部结果是无穷的，模型拟合结果显示错误数量会比其更小，因此用当前粒子逐步迭代替换
                pbest_fitness_value[i] = fitness_res[1] # 比较好的结果（错误数量）赋值给局部最优
                pbest_position[i] = particle_position_vector[i] # 这个局部最优的位置信息（参数）就是那个粒子的参数

            """
            粒子的 自身历史最优 是否替换 全局最优
            """
            if (gbest_fitness_value[1] > fitness_res[1]): # 全局的结果
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]
            elif (gbest_fitness_value[1] == fitness_res[1] and gbest_fitness_value[0] > fitness_res[0]):
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]

        '遍历每一个粒子，更新速度，位置参数'
        ''':这里需要进行一下单独的说明
        1、学习过程中参数更新，必须是整数！
        
        '''
        for i in range(args.n_particles):
            new_velocity = (args.W * velocity_vector[i]) + (args.c1 * random.random()) * (
                pbest_position[i] - particle_position_vector[i]) + (args.c2 * random.random()) *(
                    gbest_position - particle_position_vector[i])
            # print(f'粒子的更新速度：{np.ceil(new_velocity)}') # 检查用
            new_position = np.ceil(new_velocity) + particle_position_vector[i]
            particle_position_vector[i] = new_position
            # print(f'更新后的粒子位置：{particle_position_vector[i]}') # 检查用
        iteration = iteration + 1

    return gbest_position



"优化后的模型，参数加入，进行预测"
def Model_prediction(params,date,traindata,finaldata,preddata,finposition,features,files,threshold):
    '''
    :param params: 前面训练得到的最优参数
    :param traindata: 训练数据集（一般要进行process处理）
    :param preddata: 预测数据集（一般要进行process处理）
    :return:
    '''
    print(f'最佳参数n_estimators,max_depth,min_sample_leaf分别为：{params[0]}，{params[1]}，{params[2]}')
    Train_X,Train_y = finaldata
    pred_X,pred_y = preddata

    clf = RandomForestClassifier(n_estimators=params[0],
                                 max_depth=params[1],
                                 min_samples_leaf=params[2],
                                 random_state=0,
                                 n_jobs=-1)
    ##############################################################################
    "特征选择器进行特征选择"
    # RFE选择器
    Finindicators,Train_X,pred_X = Feature_selection.recursive_featres_elimination_rf(Train_X,Train_y,pred_X,clf,finposition,features)
    # RFE方案2
    # threshold = 0.0005  # 测试用
    # Finindicators, Train_X, pred_X = Feature_selection.feature_engine_rfe(Train_X, Train_y, pred_X, clf, finposition,features, threshold)
    ##############################################################################
    clf.fit(Train_X,Train_y) # 样本训练,使用全部数据训练！

    "1.模型进行测试部分"
    # train_X, test_X, train_y, test_y = train_test_split(Train_X, Train_y, random_state=0, test_size=0.3)
    # prediction_test = clf.predict(test_X)
    # pred_prob_test = clf.predict_proba(test_X)[:,1]
    # "1.1 相关得分计算"
    # scores = clf.score(test_X,test_y)
    # f1_scores = f1_score(test_y,prediction_test,average='binary')
    # auc = roc_auc_score(prediction_test,pred_prob_test)
    # print(f'PSO-RandomForest模型在样本内测试数据集的分数是：{scores}')
    # print(f'PSO-RandomForest模型在样本内测试数据集 1 scores得分为：{f1_scores}')
    # print(f'PSO-RandomForest模型在样本内测试数据集 AUC得分为：{auc}')
    # print('###########################################################')
    # with open(files,'a+') as f:
    #     f.write(f'PSO-RandomForest模型在样本内测试数据集的分数是：{scores} \n')
    #     f.write(f'PSO-RandomForest模型在样本内测试数据集 1 scores得分为：{f1_scores} \n')
    #     f.write(f'PSO-RandomForest模型在样本内测试数据集 AUC得分为：{auc} \n')
    #     f.write('########################################################### \n')
    # clf.fit(Train_X,Train_y) # 样本训练，样本外预测。如果使用的全部数据训练，这里就不用再次训练了

    "2.模型进行预测"
    prediction_pred = clf.predict(pred_X)
    pred_prob_pred = clf.predict_proba(pred_X)[:,1]
    "2.1 相关得分"
    scores = clf.score(pred_X,pred_y)
    f1_scores = f1_score(pred_y,prediction_pred,average='binary')
    fpr, tpr, thresholds = roc_curve(pred_y, prediction_pred)
    auc_scores = auc(fpr, tpr)
    precision = precision_score(pred_y,prediction_pred,average='binary')
    recall = recall_score(pred_y,prediction_pred,average='binary')
    print(f'PSO-RandomForest模型在样本外预测数据集的分数是：{scores}')
    print(f'PSO-RandomForest模型在样本外预测数据集 f1 scores得分为：{f1_scores}')
    print(f'PSO-RandomForest模型在样本外预测数据集 AUC得分为：{auc_scores}')
    print(f'PSO-RandomForest模型在样本外测试数据集 precision得分为：{precision}')
    print(f'PSO-RandomForest模型在样本外测试数据集 recall得分为：{recall}')
    print(f'PSO-RandomForest模型在 样本外测试数据集 tpr 得分为：{tpr}')
    print(f'PSO-RandomForest模型在 样本外测试数据集 fpr 得分为：{fpr}')
    print('###########################################################')
    with open(files,'a+') as f:
        f.write(f'PSO-RandomForest模型筛选后的金融指标为：{Finindicators} \n')
        f.write(f'PSO-RandomForest模型在样本外预测数据集的分数是：{scores} \n')
        f.write(f'PSO-RandomForest模型在样本外预测数据集 f1 scores得分为：{f1_scores} \n')
        f.write(f'PSO-RandomForest模型在样本外预测数据集 AUC得分为：{auc_scores} \n')
        f.write(f'PSO-RandomForest模型在样本外测试数据集 precision得分为：{precision} \n')
        f.write(f'PSO-RandomForest模型在样本外测试数据集 recall得分为：{recall} \n')
        f.write(f'PSO-RandomForest模型在 样本外测试数据集 tpr 得分为：{tpr} \n')
        f.write(f'PSO-RandomForest模型在 样本外测试数据集 fpr 得分为：{fpr} \n')
        f.write('########################################################### \n')


    "3.相关绘制方案"
    "3.1. 绘制roc，正确率和错误率图"
    # ax = plt.gca()
    # ax.set_title(f'ROC for PSO-RandomForest on {date}')
    # rfc_disp = RocCurveDisplay.from_estimator(clf, pred_X, pred_y, ax=ax,alpha=0.8)
    # plt.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/PSO_Randomforest_roc_{date}',dpi=300)
    # plt.show()

    "3.2. precison-recall"
    "绘制方案"
    # display = PrecisionRecallDisplay.from_estimator(
    #     clf, pred_X, pred_y, name="PSO-RandomForest"
    # )
    # _ = display.ax_.set_title(f"2-class Precision-Recall curve for PSO-RandomForest on {date}")
    # plt.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/PSO_Randomforest_precision_recall_{date}',dpi=300)
    # plt.show()

    "3.3 KS绘制"
    # plts = plt_ks(pred_y,pred_prob_pred,'PSO-RandomForest',date)
    # plts.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/PSO_Randomforest_KSplot_{date}',dpi=300)
    "3.4 机器学习可解释性分析"
    "shap_value: 创建这个解释器"
    # explainer = shap.TreeExplainer(clf)
    # shape_value = explainer.shap_values(pred_X)
    ""
    # shap.initjs()
    # shap.force_plot(explainer.expected_value[1],shape_value[1],pred_X)

    "Advanceed shape value:"
    # shap.summary_plot(shape_value[1],pred_X,title=f'SHAP value of PSO-RandomForest on date {date}')
    # plt.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/PSO_Randomforest_shap_summary_{date}',dpi=300)
    # plt.show()

    return clf




if __name__ == '__main__':
    pass