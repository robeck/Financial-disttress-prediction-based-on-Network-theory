'''
@Author: haozhi chen
@Date: 2022-11
@Target: This class use RFE method to preselect the important features from baseline datasets

'''

import numpy as np
import pandas as pd
import random
import shap
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_selection import RFECV,RFE  # recursive feature elimination (REF) 递归特征消除

from Model.PSO_SVM.Config import args, kernel
from Model.PSO_SVM.utile import confusion_martrix_disp, plot
from Results_Plot.KS_curve import plt_ks

"模型评估和数据绘制"
from sklearn.metrics import confusion_matrix, RocCurveDisplay, f1_score, PrecisionRecallDisplay, \
    roc_auc_score  # ROC-AUC曲线

import matplotlib.pyplot as plt

'''SVM模型拟合过程(适应度函数）
1）拟合
2）绘制混淆矩阵
3）输出混淆矩阵结果
'''


def fittess_function(params, data):
    '''
    :param params:
    :param data:

    :return confusion matrix:
        [0] 训练集 的预测错误结果统计
        [1] 测试集 的预测错误结果统计
    '''
    train_x, train_y, test_x, test_y = data  # 返回的是多个数据，可以直接这样赋值
    classifer = make_pipeline(StandardScaler(),
                              SVC(kernel=kernel, gamma=params[0], C=params[1], max_iter=2000, random_state=66,
                                  probability=True))
    classifer.fit(train_x, train_y)
    y_train_pred = classifer.predict(train_x)
    y_test_pred = classifer.predict(test_x)

    "绘制以下测试数据集的 混淆矩阵"
    # label_names = ['label1','label2']
    # titles = [("confusion matrix without norm",None),
    #           ("confusion matrix with noem","true")]
    # confusion_martrix_disp(classifer,test_x,test_y,label_names,titles)

    return confusion_matrix(train_y, y_train_pred)[0][1] + confusion_matrix(train_y, y_train_pred)[1][0], \
           confusion_matrix(test_y, y_test_pred)[0][1] + confusion_matrix(test_y, y_test_pred)[1][0]


def pso_svm_model(data):
    # 初始化参数
    iteration = 0  # 初始迭代标记
    '''
    参数1：代表粒子位置，其实也是适应函数输入的参数（gamma，c）
    其中random.random()*10意味着取值范围在[0-10]之间！
    '''
    particle_position_vector = np.array(
        [np.array([random.random() * 10, random.random() * 10]) for _ in range(args.n_particles)])  # 初始化每一个粒子的位置
    # 参数2：粒子自身历史最优位置
    pbest_position = particle_position_vector
    # 参数3：粒子自身最优的适应函数值 初始化为 inf
    pbest_fitness_value = np.array([float('inf') for _ in range(args.n_particles)])
    # 参数4，5：全局位置初始，全局适应函数初始化
    gbest_fitness_value = np.array([float('inf'), float('inf')])
    gbest_position = np.array([float('inf'), float('inf')])
    # 参数6：速度向量初始化
    velocity_vector = ([np.array([0, 0]) for _ in range(args.n_particles)])

    '进行不断的迭代'
    while iteration < args.n_iterations:
        print(f'迭代的轮数为：{iteration}')
        # plot(particle_position_vector) # 绘制初始化的粒子分布散点图
        '遍历100个粒子'
        for i in tqdm(range(args.n_particles)):
            fitness_res = fittess_function(particle_position_vector[i], data)  # 统计预测结果
            # print("error of priticle ",i,'is (training,test)',fitness_res,"At (gamma,c): ",
            #       particle_position_vector[i]) # 参数的输出显示

            """
            初始化的 自身历史最优 进行迭代替换
            （1）比较
            （2）用较好结果 替换 自身历史最优：这是一个自己比较的过程
            （3）粒子位置信息（参数）替换
            """
            if (pbest_fitness_value[i] > fitness_res[1]):  # 因为初始的局部结果是无穷的，模型拟合结果显示错误数量会比其更小，因此用当前粒子逐步迭代替换
                pbest_fitness_value[i] = fitness_res[1]  # 比较好的结果（错误数量）赋值给局部最优
                pbest_position[i] = particle_position_vector[i]  # 这个局部最优的位置信息（gamma，c 参数）就是那个粒子的参数

            """
            粒子的 自身历史最优 是否替换 全局最优
            """
            if (gbest_fitness_value[1] > fitness_res[1]):  # 全局的结果
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]
            elif (gbest_fitness_value[1] == fitness_res[1] and gbest_fitness_value[0] > fitness_res[0]):
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]

        '遍历每一个粒子，更新速度，位置参数'
        for i in range(args.n_particles):
            new_velocity = (args.W * velocity_vector[i]) + (args.c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (args.c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position

        iteration = iteration + 1

    return gbest_position


def best_model(params, date, datatrain, datapred):
    sc = StandardScaler()
    TrainX,TrainY = np.array(datatrain.iloc[:,:-1]),np.array(datatrain.iloc[:,-1:]).ravel()
    PredX,PredY = np.array(datapred.iloc[:,:-1]),np.array(datapred.iloc[:,-1:]).ravel()
    "数据标准化 Standscaler"
    stdTrainX= sc.fit_transform(TrainX)
    stdPredX = sc.fit_transform(PredX)

    "提取financial部分，nonfinancial部分的数据"
    FinTrainX,NonfinTrainX = stdTrainX[:,5:],stdTrainX[:,0:5] # 金融指标部分，非金融指标部分

    print(f'The best parameters in current work gamma: {params[0]}, C: {params[1]}')
    "1. 构建模型，筛选数据即，拟合"
    '''
    基本思路是：
    1）先使用基础的SVC来过滤有效的变量，参数
    2）使用优化过参数的模型，用SVC过滤的数据集进行操作
    '''
    classifer = SVC(C=1,kernel='linear',probability=True) # 用于筛选数据的SVC模型
    selector = RFE(estimator=classifer,step=1)
    selector.fit(FinTrainX,TrainY) # 从金融指标 <-> target中 找到合适的指标！
    NewfintrainX = selector.transform(FinTrainX) # Reduce X to the selected features 形成新的features
    print(f'我们对金融特征部分进行选择其中的 mask of selected features {selector.support_},特征的排序 {selector.ranking_}')
    "1.1. 组装数据，切割"
    TrainX = np.c_[NewfintrainX,NonfinTrainX] # 训练数据X
    trainX,testX,trainy,testy = train_test_split(TrainX,TrainY,random_state=10,test_size=0.3) # 根据筛选好的数据重新分割
    "1.2 重新构建，训练训练模型"
    classifer = SVC(kernel=kernel, gamma=params[0], C=params[1], max_iter=2000, random_state=66,
                    probability=True)  # 非pipline模型
    classifer.fit(TrainX,TrainY) #整体训练！

    "2. 样本内测试数据的检验" # 测试可以单独测试
    y_test_pred = classifer.predict(testX)
    y_test_prob = classifer.predict_proba(testX)[:, 1]
    "样本内测试数据得分情况"
    scores_test_pred = classifer.score(testX, testy)
    f1_scores = f1_score(testy, y_test_pred, average='binary')
    auc_scores = roc_auc_score(testy, y_test_prob)
    print(f'PSO-RFE-SVM模型在测试数据中的分数：{scores_test_pred}')
    print(f'PSO-RFE-SVM模型在测试数据集中的f1 score得分为：{f1_scores}')
    print(f'PSO-RFE-SVM模型在测试数据集中的AUC得分为：{auc_scores}')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_RFE_SVM_res.txt', 'a+') as f:
        f.write(f'PSO-RFE-SVM模型在测试数据中的分数：{scores_test_pred} \n')
        f.write(f'PSO-RFE-SVM模型在测试数据集中的f1 score得分为：{f1_scores} \n')
        f.write(f'PSO-RFE-SVM模型在测试数据集中的AUC得分为：{auc_scores} \n')

    "3. 样本外预测集的检验"
    # y_pred_pred = classifer.predict(predX)
    # y_pred_prob = classifer.predict_proba(predX)[:, 1]  # 计算X中样本的可能结果的概率
    # # print(f'预测的结果为：{y_pred_pred}')
    # "各项预测分数"
    # score_pred_pred = classifer.score(predX, predy)
    # f1_scores = f1_score(predy, y_pred_pred, average='binary')
    # auc_scores = roc_auc_score(predy, y_pred_prob)
    # print(f'PSO-SVM模型在预测数据集中的分数为：{score_pred_pred}')
    # print(f'PSO-SVM模型在预测数据集中的f1 scores得分为：{f1_scores}')
    # print(f'PSO-SVM模型在预测数据集中的AUC得分为：{auc_scores}')
    # with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_SVM_res.txt', 'a+') as f:
    #     f.write(f'PSO-SVM模型在预测数据集中的分数为：{score_pred_pred} \n')
    #     f.write(f'PSO-SVM模型在预测数据集中的f1 scores得分为：{f1_scores} \n')
    #     f.write(f'PSO-SVM模型在预测数据集中的AUC得分为：{auc_scores} \n')

    "4. 绘制预测，实际值"
    # plt.plot(predy,color='red',label='true')
    # plt.plot(y_pred_pred,color='blue',label='pred')
    # plt.show()
    # "roc绘制"
    # ax = plt.gca()
    # ax.set_title(f'ROC for PSO-SVM on {date}')
    # svc_disp = RocCurveDisplay.from_estimator(classifer, predX, predy, ax=ax, alpha=0.7)
    # plt.show()
    # "precison-recall"
    # "绘制方案"
    # display = PrecisionRecallDisplay.from_estimator(
    #     classifer, predX, predy, name="PSO-SVM"
    # )
    # _ = display.ax_.set_title(f"PSO-SVM 2-class Precision-Recall curve for PSO-SVM on {date}")
    # plt.show()
    # "KS绘制"
    # plt_ks(predy, y_pred_prob, 'PSO-SVM', date)  # 绘制KS 测试一下！
    # "SHAP value的绘制工作"
    # explainer = shap.KernelExplainer(classifer.predict_proba, trainX, link='logit')
    # # shap.DeepExplainer 为深度学习模型工作
    # # shap.KernelExplainer 为所有模型工作
    # # shap.TreeExplainer 为决策树，随机森林工作
    # shap_values = explainer.shap_values(predX)
    # shap.summary_plot(shap_values[1], predX, title=f'SHAP value of PSO-SVM on {date} ')
    # plt.show()

    return None


"测试用"

if __name__ == '__main__':
    pass