'''
@Author: haozhi chen
@Date: 2022-09
@Target: 针对DEA计算中存在某些数据无法得到结果的问题！我们必须对数据进行有效性的筛选！这样才更加具有可靠性和意义！

'''
import pandas as pd
import numpy as np
import logging
import os
import time
from Model import RandomForest,GS_SVM
from Model.PSO_RandomForest import PSO_RandomForest,Adaboost_RandomForest,Adaboost_PSO_RandomForest
from Model.Comparasion_model import data_preprocess,logit_,logistic_,svc_,Navbay_,gbdt_
from Model.GBDT_LR import GBDT # GBDT生成混合新特征参数的部分
from Model.PSO_SVM import Input_data,pso_svm,Adaboost_PSO_SVM,Bagging_PSO_SVM # PSO_SVM
from Workshop.Main_feature_data_process import multiprocess_merge
from Dataprocess import Risk_stock_process
from multiprocessing import Process #多进程
from multiprocessing import Pool # 多进程池


"calibration"
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from Model.Comparasion_model import NaivelyCalibratedLinearSVC
from Results_Plot import Main_plot

# 配置logger ########################################################################
# 为了保证每一次这个log文件都是全新的，我们需要做一下额外的配置，检查->存在即删除
if os.path.exists(r'/home/haozhic2/ListedCompany_risk/Data/MainFunction_log.txt'):
    os.remove(r'/home/haozhic2/ListedCompany_risk/Data/MainFunction_log.txt')
else:
    pass
"logger配置"
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f'/home/haozhic2/ListedCompany_risk/Data/MainFunction_log.txt')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
####################################################################################


# "全局变量，对数据中fin指标位置，筛选特征个数进行调整"
# finposition = 4 # 金融指标的起始位置只有中心性时候是 4；全部网络指标的时候 6；如果没有网络指标的话设置为 0
# features = 8 # 选择features的个数

###################################################################################

"运行结果的主函数"
''':param
start_date,end_date : 这是调整预测周期，时间，以及多长时间的样本进行学习的变量
call_datatags ：到低要不要召唤DEA数据，目前不需要
'''
def main(round,modeltype,threshold):
    # 初始变量和参数
    start_date, end_date = 2011, 2022
    years = [year for year in range(start_date, end_date)] # 年份时间的list
    ###############################################################################################
    "读取数据时候，选择的数据标签"
    # call_datatags = 'full' # 使用标记欠缺数据，做为最终数据标签的
    # call_datatags = 'justed_financial'  # 只使用全部的金融指标！
    call_datatags = 'adjusted_network'  # 筛选了部分网络指标如：只有Centrality
    # call_datatags = 'adjusted_financial'  # 筛选了部分金融指标，网络指标位置参数为5
    # call_datatags = 'justed_financial'  # 只使用筛选了部分的金融指标
    # call_datatags = 'adjusted_network_financial'  # 使用调整了网络，金融的指标 网络指标位置参数为5
    ##############################################################################################
    "控制读取new网络 还是旧网络的数据"
    nettags = 'new' # 新网络的数据
    # nettags = '' # 不使用新网络
    #############################################################################################
    "选择的参数个数keynum"
    keynum = 8 # 12,18,


    "0. 数据特征的汇总，我们在Main_feature_data_process中进行，读取该程序运行结果输出的文件即可,也可以再次根据需求运行"
    # integrated_data_clean = pd.read_csv('~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data.csv') # 读取最初始汇总的全部数据
    integrated_data_clean = multiprocess_merge(start_date, end_date,call_datatags,nettags) # 读取必要针对性处理后的数据（比如调整DEA方法等）
    print(integrated_data_clean.columns)

    ###############################################################################################################
    "1. 滚动预测部分"
    dateinterval = 3 # 时期的间隔为2，或者为3 均可测试
    windowsize = 1 # 时间的滚动窗口长度为1
    arglists = []
    for date in range(start_date,end_date-dateinterval,windowsize): # 结尾位置 -dateinterval，实现的是最后一起3年直接预测
        # 变量初始化
        preddate = date + dateinterval

        print(f'the current working date range is {date} to {preddate}')
        start_date,middle_date,end_date = date,date+dateinterval-windowsize,date+dateinterval
        #####################################################
        "滚动预测方案2:2011-2014,2011-2015,2011-2016，。。。"
        # start_date = 2011 # 固定住起始时间
        #####################################################
        "进行预测的数据需要调整，处理"
        traingdataframe = integrated_data_clean[(integrated_data_clean['Date']>=start_date) & (integrated_data_clean['Date']<=middle_date)].drop(['Stkcd','Date'],axis=1) # 剔除id，日期
        preddataframe = integrated_data_clean[integrated_data_clean['Date']==end_date].drop(['Stkcd','Date'],axis=1)
        # print(traingdataframe.head(20))  # 输出预测数据集的前20个公司id，实际风险标签 ST_value
        # print(traingdataframe.tail(20))

        "1.1 Multiprocess runing 对不同的模型"
        '''
        1)这里进行循环，每次设置一个fin feature的个数
        2)这个数继续被使用到进行结果的标记中
        '''
        "全局变量，对数据中fin指标位置，筛选特征个数进行调整"
        finposition = 4  # 金融指标的起始位置只有中心性时候是 4；全部网络指标的时候 6；如果没有网络指标的话设置为 0
        features = keynum  # 选择features的个数
        "1.1.1 参数设定"
        arglist = (traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round,threshold) # 参数设定
        arglists.append(arglist)
        "1.1.2 多模型的设定和多进程的运行"
        if modeltype == 'multiple':
            # models = None #初始化
            # models = [RandomForest_process,Adaboost_RandomForest_process,PSO_RandomForest_process,
            #           Adaboost_PSO_RandomForest_process,PSO_SVM_process,Adaboost_PSO_SVM_process,Bagging_PSO_SVM_process] # 全模型
            models = [RandomForest_process,Adaboost_RandomForest_process,PSO_RandomForest_process,Adaboost_PSO_RandomForest_process] # randomforest 模型
            # models = [PSO_SVM_process,Adaboost_PSO_SVM_process,Bagging_PSO_SVM_process] # SVM 模型
            # models = [Adaboost_RandomForest_process, Adaboost_PSO_RandomForest_process]  # randomforest 部分模型

            for i,model in enumerate(models):
                process = Process(target=model,args=arglist)
                print(f'process {i} are working')
                process.start()
                time.sleep(5)
                # process.join()
            pass

        elif modeltype == 'single':
            # RandomForest_process(arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5],arglist[6])
            # Adaboost_RandomForest_process(arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5],arglist[6],arglist[7])
            PSO_RandomForest_process(arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5],arglist[6],arglist[7])
            # Adaboost_PSO_RandomForest_process(arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5],arglist[6],arglist[7])
            # PSO_SVM_process(arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5],arglist[6],arglist[7])
            # Adaboost_PSO_SVM_process(arglist[0],arglist[1],arglist[2])
            # Bagging_PSO_SVM_process(arglist[0],arglist[1],arglist[2])
            # GridSearch_SVC_process(arglist[0],arglist[1],arglist[2]) #grid search 的svc
            "1.2 single running 单个按照顺序运行"
        else:
            pass


        "2. 比较模型！: "
        "logit"
        # std_tag='Max'
        # lg = logit_(data_preprocess(traingdataframe,preddataframe,std_tag)) # features太多，矩阵歧义了
        "logistic"
        # std_tag = False
        # lr = logistic_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "nb"
        # std_tag = False
        # gnb = Navbay_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "SVC"
        # std_tag = True
        # svc = svc_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "GBDT"
        # std_tag = False
        # gbdt = gbdt_(data_preprocess(traingdataframe, preddataframe, std_tag))

        "3. 功能性增强"
        "绘制1：calibration curve"
        # lr = LogisticRegression()
        # gnb = GaussianNB()
        # svc = NaivelyCalibratedLinearSVC(C=1.0)
        # rfc = RandomForestClassifier()
        # clflist = [(lr,'logistic',False),(gnb,'Naive Bayes',False),(svc,'SVC',True),(rfc,"Random Forest",False)]
        # Main_plot.calibration_plot(clflist,traingdataframe,preddataframe)

        "绘制2：结果的可解释性：这部分可以直接在模型中运行"
        # PDP 绘制

    return None

"考虑多进程工作！逐个唤起不同的模型"
"Model1: RandomForest"
def RandomForest_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags, round,threshold):
    print('####################### 当前运行的模型的RandomForest ######################')
    files = f'/home/haozhic2/ListedCompany_risk/Results_output/Source_file/RandomForest_res_{call_datatags}_{features}_fin_round{round}.txt'
    with open(files, 'a+') as f:
        f.write(f'这是随机森林模型，工作于{preddate} \n')
    RF_Model = RandomForest.RF_model(
        RandomForest.input_data_process(traingdataframe, 'train_test'),  # 进行模型的训练和测试
        RandomForest.input_data_process(traingdataframe, 'normal'),  # 使用全部数据的训练
        RandomForest.input_data_process(preddataframe, 'normal'),  # 预测的数据
        finposition,  # 金融指标的位置
        features,  # 筛选特征的个数
        files, # 存储的文件
        threshold
    )  # 生成自己的model，训练数模型，进行预测，一次性完成

    return None

"Model2: Adaboost_RandomForest"
def Adaboost_RandomForest_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round,threshold):
    print('####################### 当前运行的模型的Adaboost_RandomForest ######################')
    files = f'/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_RandomForest_res_{call_datatags}_{features}_fin_round{round}.txt'
    with open(files, 'a+') as f:
        f.write(f'这是随机森林模型，工作于{preddate} \n')
    RF_Model = Adaboost_RandomForest.RF_model(
        Adaboost_RandomForest.input_data_process(traingdataframe, 'train_test'),  # 进行模型的训练和测试
        Adaboost_RandomForest.input_data_process(traingdataframe, 'normal'), # 使用全部数据的训练
        Adaboost_RandomForest.input_data_process(preddataframe, 'normal'),  # 预测的数据
        finposition,  # 金融指标的位置
        features,  # 筛选特征的个数
        files,
        threshold
    )  # 生成自己的model，训练数据集内的测试

    return None

"Model3: PSO-RandomForest"
def PSO_RandomForest_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round,threshold):
    print('####################### 当前运行的模型的PSO-RandomForest #######################')
    files = f'/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_RandomForest_res_{call_datatags}_{features}_fin_round{round}.txt'
    with open(files, 'a+') as f:
        f.write(f'这是PSO优化的随机森林模型，工作于{preddate} \n')
    best_parameters = PSO_RandomForest.PSO_RF_Model(PSO_RandomForest.input_data_process(traingdataframe, 'train_test'))
    PSO_RandomForest_model = PSO_RandomForest.Model_prediction(
        best_parameters,  # 模型参数
        preddate,  # 模型运行的日期
        PSO_RandomForest.input_data_process(traingdataframe, 'train_test'),  # 拟合的数据
        PSO_RandomForest.input_data_process(traingdataframe, 'normal'),  # 最终的训练数据
        PSO_RandomForest.input_data_process(preddataframe, 'normal'),  # 预测的数据
        finposition, # 金融指标的位置
        features, # 筛选特征的个数
        files,
        threshold
    )
    return None

"Model4: Adaboost-PSO-RandomForest"
def Adaboost_PSO_RandomForest_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round,threshold):
    print('####################### 当前运行的模型的Adaboost-PSO-RandomForest #######################')
    files = f'/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_PSO_RandomForest_res_{call_datatags}_{features}_fin_round{round}.txt'
    with open(files, 'a+') as f:
        f.write(f'这是PSO优化的ADABOOST-随机森林模型，工作于{preddate} \n')
    best_parameters = Adaboost_PSO_RandomForest.PSO_RF_Model(Adaboost_PSO_RandomForest.input_data_process(traingdataframe, 'train_test'))
    Adaboost_PSO_RandomForest_model = Adaboost_PSO_RandomForest.Model_prediction(
        best_parameters,  # 模型参数
        preddate,  # 模型运行的日期
        Adaboost_PSO_RandomForest.input_data_process(traingdataframe, 'train_test'),  # 拟合的数据
        Adaboost_PSO_RandomForest.input_data_process(traingdataframe, 'normal'),  # 最终的训练数据
        Adaboost_PSO_RandomForest.input_data_process(preddataframe, 'normal'),  # 预测的数据
        finposition,  # 金融指标的位置
        features,  # 筛选特征的个数
        files,
        threshold # 测试用
    )
    return None


"Model5： PSO-SVM"
def PSO_SVM_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round):
    "数据处理部分：参数使用了标准化！"
    train_testdata = Input_data.data_process1(traingdataframe)  # 这个输出有四个参数，直接使用即可!注意这个参数的顺序！！
    train_data = Input_data.data_process2(traingdataframe) # 输出两个参数，用于整体的训练
    pred_data = Input_data.data_process2(preddataframe)  # 输出两个参数，进行预测
    "模型生成部分"
    print('####################### 当前运行的模型师PSO-SVM #######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_SVM_res.txt', 'a+') as f:
        f.write(f'这是PSO-SVM模型工作于{preddate} \n')
    best_params = pso_svm.pso_svm_model(train_testdata)  # pso-svm输出的最佳参数
    pso_svm.best_model(best_params, preddate, train_data,pred_data,finposition,features)
    return None


"Model6: 是用Adaboost-Random-forest 选择金融特征，是用Adaboost-PSO-SVM 模型进行预测"
''':说明
1）如果要调整特征选择，金融模型中标注RandomForest部分即可
'''
def Adaboost_PSO_SVM_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round):
    train_testdata = Input_data.data_process1(traingdataframe)  # 这个输出有四个参数，直接使用即可!注意这个参数的顺序！！
    train_data = Input_data.data_process2(traingdataframe)  # 输出两个参数，用于整体的训练
    pred_data = Input_data.data_process2(preddataframe)  # 输出两个参数，进行预测
    "模型生成部分"
    print('###################### 当前运行的模型是Adaboost_PSO_SVM_process ######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/Adaboost_PSO_SVM_res.txt','a+') as f:
        f.write(f'这是Adaboost-PSO-SVM模型工作于{preddate} \n')
    best_params = Adaboost_PSO_SVM.pso_svm_model(train_testdata)
    Adaboost_PSO_SVM.best_model(best_params,preddate,train_data,pred_data,finposition,features)

"Model7: Bagging-forest 选择金融特征，是用Bagging-PSO-SVM 模型进行预测"
''':说明
1）如果要调整特征选择，金融模型中标注RandomForest部分即可
'''
def Bagging_PSO_SVM_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round):
    train_testdata = Input_data.data_process1(traingdataframe)  # 这个输出有四个参数，直接使用即可!注意这个参数的顺序！！
    train_data = Input_data.data_process2(traingdataframe)  # 输出两个参数，用于整体的训练
    pred_data = Input_data.data_process2(preddataframe)  # 输出两个参数，进行预测
    "模型生成部分"
    print('###################### 当前运行的模型是Adaboost_PSO_SVM_process ######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/bagging_PSO_SVM_res.txt','a+') as f:
        f.write(f'这是Bagging-PSO-SVM模型工作于{preddate} \n')
    best_params = Bagging_PSO_SVM.pso_svm_model(train_testdata)
    Bagging_PSO_SVM.best_model(best_params,preddate,train_data,pred_data,finposition,features)


'''说明
1）SVC方法存在一个问题：参数选择的时候模型C参数必须等于1
'''
"grdisearch 的 svc 模型"
def GridSearch_SVC_process(traingdataframe,preddate,preddataframe,finposition,features,call_datatags,round):
    train_data = Input_data.data_process2(traingdataframe)  # 输出两个参数，用于整体的训练
    pred_data = Input_data.data_process2(preddataframe)  # 输出两个参数，进行预测
    "模型生成部分"
    print('###################### 当前运行的模型是Gridsearch_SVC_process ######################')
    # 输出日志部分
    # 模型调用
    GS_SVM.gridsearch_svc(preddate,train_data,pred_data)

if __name__ == '__main__':
    models = 'single' # 'single','multiple' 单个模型还是多模型多进程,'None' 为比较模型
    threshold = 0.0005 # 0.0005， 0.0016
    rounds = 10
    # rounds = 5
    # round = 1
    "单个main操作"
    # main(round,models,threshold)


    "多进程方法1"
    # 进程池
    # pool = Pool(rounds)
    # # 启动多进程
    # pool.map(main,range(rounds))
    # # colse
    # pool.close()
    # # close
    # pool.join()
    "多进程方法2"
    # Threshold = list(np.linspace(0.0005,0.01,10))
    # threshold = Threshold[1]
    for i,round in enumerate(range(rounds)):
        # threshold = Threshold[i]
        arg = (round,models,threshold) # 转化list
        print(arg)
        process = Process(target=main, args=arg)
        print(f'process round {round} are working')
        process.start()
        time.sleep(5)
        # process.join()

