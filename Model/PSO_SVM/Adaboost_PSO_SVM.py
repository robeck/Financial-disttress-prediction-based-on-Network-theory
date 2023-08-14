'''
@Author: haozhi chen
@Date: 2022-08
@Target: 实现以下基于粒子群优化的SVM模型

特点和注意事项！
（1）目前测试的是对参数 gamma，C 的优化。如需要调整则对 particle_position_vector 进行变化即可

2023-02:
（1）实现对Financial indicators的 选择 是用RandomForest算法
（2）实现Adaboost 算法 来 优化PSO-SVM魔心
'''
import numpy as np
import pandas as pd
import random
import shap
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier # adaboost算法
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import RFECV  # recursive feature elimination (REF) 递归特征消除

from Model import Feature_selection
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


def best_model(params, date, traindata, preddata,finposition,features):
    sc = StandardScaler()
    TrainX, Trainy = traindata
    predx, predy = preddata
    "数据标准化 Standscaler"
    TrainX = sc.fit_transform(TrainX)
    predX = sc.fit_transform(predx)

    "*是用RandomForest进行特征选择的过程"
    # TrainX,predX = Feature_selection.selection_process(TrainX,Trainy,predX,finposition,features) #模块化调用，便于随时变换！
    "*使用RFE进行特征选择的过程"
    # TrainX,predX = Feature_selection.recursive_featres_elimination_svc(TrainX, Trainy, predX,params,finposition,features) # 模块化，随时可调换

    # 划分数据集
    trainX,testX,trainy,testy = train_test_split(TrainX, Trainy,random_state=10,test_size=0.3) # 简单的生成一个测试集和
    print(f'The total features are {TrainX.shape[1]}')
    ###############################################################################
    print(f'The best parameters in current work gamma: {params[0]}, C: {params[1]}')
    ################################################################################
    "1. 构建模型，拟合"
    '''
    Pipeline: 适用于数据中没有binary变量
    非Pipeline：适用于任何数据！
    '''
    # SVM 基学习器
    clf = SVC(kernel=kernel, gamma=params[0], C=params[1], max_iter=2000, random_state=66,
              probability=True)  # 非pipline模型
    # Adaboost 学习器
    adaboost = AdaBoostClassifier(base_estimator=clf,n_estimators=50,random_state=66)

    "K折交叉验证-输出结果"
    # scores = cross_val_score(adaboost,TrainX,Trainy,cv=5)
    # with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_PSO_SVM_res.txt', 'a+') as f:
    #     f.write(f'PSO-SVM模型交叉验证的平均得分为：{scores.mean()} \n')

    # 训练模型
    clf.fit(TrainX, Trainy)  # 一体化的训练！保留一个clf的训练
    adaboost.fit(TrainX, Trainy) #
    ##################################################################################

    "2. 样本内测试数据的检验"
    y_test_pred = adaboost.predict(testX)
    y_test_prob = adaboost.predict_proba(testX)[:, 1]
    "样本内测试数据得分情况"
    scores_test_pred = adaboost.score(testX, testy)
    f1_scores = f1_score(testy, y_test_pred, average='binary')
    auc_scores = roc_auc_score(testy, y_test_prob)
    print(f'Adaboost-PSO-SVM模型在测试数据中的分数：{scores_test_pred}')
    print(f'Adaboost-PSO-SVM模型在测试数据集中的f1 score得分为：{f1_scores}')
    print(f'Adaboost-PSO-SVM模型在测试数据集中的AUC得分为：{auc_scores}')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_PSO_SVM_res.txt', 'a+') as f:
        f.write(f'Adaboost-PSO-SVM模型在测试数据中的分数：{scores_test_pred} \n')
        f.write(f'Adaboost-PSO-SVM模型在测试数据集中的f1 score得分为：{f1_scores} \n')
        f.write(f'Adaboost-PSO-SVM模型在测试数据集中的AUC得分为：{auc_scores} \n')

    "3. 样本外预测集的检验"
    "进行预测"
    y_pred_pred = adaboost.predict(predX)
    y_pred_prob = adaboost.predict_proba(predX)[:, 1]  # 计算X中样本的可能结果的概率
    # print(f'预测的结果为：{y_pred_pred}')
    "各项预测分数"
    score_pred_pred = adaboost.score(predX, predy)
    f1_scores = f1_score(predy, y_pred_pred, average='binary')
    auc_scores = roc_auc_score(predy, y_pred_prob)
    print(f'Adaboost-PSO-SVM模型在预测数据集中的分数为：{score_pred_pred}')
    print(f'Adaboost-PSO-SVM模型在预测数据集中的f1 scores得分为：{f1_scores}')
    print(f'Adaboost-PSO-SVM模型在预测数据集中的AUC得分为：{auc_scores}')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_PSO_SVM_res.txt', 'a+') as f:
        f.write(f'Adaboost-PSO-SVM模型在预测数据集中的分数为：{score_pred_pred} \n')
        f.write(f'Adaboost-PSO-SVM模型在预测数据集中的f1 scores得分为：{f1_scores} \n')
        f.write(f'Adaboost-PSO-SVM模型在预测数据集中的AUC得分为：{auc_scores} \n')

    "4. 绘制预测，实际值"
    # plt.plot(predy,color='red',label='true')
    # plt.plot(y_pred_pred,color='blue',label='pred')
    # plt.show()
    "roc绘制"
    # ax = plt.gca()
    # ax.set_title(f'ROC for Adaboost-PSO-SVM on {date}')
    # svc_disp = RocCurveDisplay.from_estimator(adaboost, predX, predy, ax=ax, alpha=0.7)
    # plt.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/Adaboost_PSO_SVM_roc_{date}', dpi=300)
    # plt.show()

    "precison-recall"
    "绘制方案"
    # display = PrecisionRecallDisplay.from_estimator(
    #     adaboost, predX, predy, name="Adaboost-PSO-SVM"
    # )
    # _ = display.ax_.set_title(f" 2-class Precision-Recall curve for Adaboost-PSO-SVM on {date}")
    # plt.savefig(
    #     f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/Adaboost_PSO_SVM_precision_recall_{date}',
    #     dpi=300)
    # plt.show()

    "KS绘制"
    # plts = plt_ks(predy, y_pred_prob, 'Adaboost-PSO-SVM', date)  # 绘制KS 测试一下！
    # plts.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/Adaboost_PSO_SVM_KSplot_{date}', dpi=300)

    "SHAP value的绘制工作"
    # explainer = shap.KernelExplainer(adaboost.predict_proba, shap.kmeans(trainX,10), link='logit') # 增加shap.kmeans，加快速度
        # shap.DeepExplainer 为深度学习模型工作
        # shap.KernelExplainer 为所有模型工作
        # shap.TreeExplainer 为决策树，随机森林工作
    # shap_values = explainer.shap_values(predX)
    # shap.summary_plot(shap_values[1], predX, title=f'SHAP value of Adaboost-PSO-SVM on {date} ')
    # plt.savefig(f'/home/haozhic2/ListedCompany_risk/Results_output/Figure_file/Adaboost_PSO_SVM_shap_summary_{date}',
    #             dpi=300)
    # plt.show()

    return None


"测试用"

if __name__ == '__main__':
    pass