'''
@Author: haozhi chen
@Date: 2022-09
@Target: 针对DEA计算中存在某些数据无法得到结果的问题！我们必须对数据进行有效性的筛选！这样才更加具有可靠性和意义！

'''
import pandas as pd
import logging
import os
import time
from Model import RandomForest
from Model.PSO_RandomForest import PSO_RandomForest
from Model.Comparasion_model import data_preprocess,logit_,logistic_,svc_,Navbay_
from Model.GBDT_LR import GBDT # GBDT生成混合新特征参数的部分
from Model.PSO_SVM import Input_data,pso_svm,RFE_PSO_SVM # PSO_SVM
from Workshop.Main_feature_data_process import financial_data_input,intege_two
from Dataprocess import Risk_stock_process
from multiprocessing import Process #多进程

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

def main():
    # 初始变量和参数
    start_date, end_date = 2015, 2022
    years = [year for year in range(start_date, end_date)] # 年份时间的list
    ###############################################################################################
    # deatags = ['financial','innovation','growth','operation','total'] # 多组结果测试用
    deatags = 'total'  # DEA计算哪些数据的标签
    dea_method = 'CRS'  # DEA方法的标签
    networkcuttag = 'Cut_zero'  # 网络缩减的标签

    "0. 考虑一次性汇总全部数据，因为后续的滚动会有重叠日期，数据尽量一次性汇总和调用"
    '''
    1) 仅合并了金融数据
    2）作为测试数据而已
    '''
    # integrated_data = pd.DataFrame()  # 存储汇总好的数据
    # for date in range(start_date, end_date):
    #     print(f'-----------------The current intergrate research data on {date}--------------------')
    #     "读取有，无风险标识的股票"
    #     df_ST, df_nonST, df_stocktags = Risk_stock_process.stocklist(date)
    #
    #     "3) 金融预测的financial数据提取"
    #     financialres = financial_data_input(df_stocktags, date)
    #     "仅金融数据合并"
    #     temp_integredata = intege_two(df_stocktags, financialres)  # 合并研究标签，特征数据（测试）
    #     ######################################################################
    #
    #     temp_integredata['Date'] = date  # 这个必不可少，因为df_stocktags中没有date日期
    #     "不同日期下的 final数据 的纵向合并"
    #     integrated_data = pd.concat([integrated_data, temp_integredata], axis=0)
    #
    # "数据需要清理nan数据，那么有两种方案：1）fill，2）drop"
    # print(integrated_data.head())
    # print(integrated_data.shape)
    # integrated_data_clean = integrated_data.dropna()  # 删除nan存在的行数据
    # print(integrated_data_clean.shape)
    ###############################################################################################################

    "0. 数据特征的汇总，我们在Main_feature_data_process中进行，读取该程序运行结果输出的文件即可"
    integrated_data_clean = pd.read_csv('~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data.csv')

    "1. 滚动预测部分"
    dateinterval = 2 # 时期的间隔为2
    windowsize = 1 # 时间的滚动窗口长度为1
    for date in range(start_date,end_date-dateinterval,windowsize): # 结尾位置 -dateinterval，实现的是最后一起3年直接预测
        # 变脸初始化
        preddate = date + dateinterval

        print(f'the current working date range is {date} to {preddate}')
        start_date,middle_date,end_date = date,date+1,date+dateinterval
        "进行预测的数据需要调整，处理"
        traingdataframe = integrated_data_clean[(integrated_data_clean['Date']==start_date) | (integrated_data_clean['Date']==middle_date)].drop(['Stkcd','Date'],axis=1) # 剔除id，日期
        preddataframe = integrated_data_clean[integrated_data_clean['Date']==end_date].drop(['Stkcd','Date'],axis=1)
        # print(traingdataframe.head(20))  # 输出预测数据集的前20个公司id，实际风险标签 ST_value
        # print(traingdataframe.tail(20))

        "1.1 Multiprocess runing 对不同的模型"
        arglist = (traingdataframe,preddate,preddataframe)
        # models =[RandomForest_process,PSO_RandomForest_process,PSO_SVM_process]
        models = [PSO_RFE_SVM_process]
        # models = [RandomForest_process,PSO_RandomForest_process,PSO_SVM_process,PSO_RFE_SVM_process] # 全模型
        for i,model in enumerate(models):
            process = Process(target=model,args=arglist)
            process.start()
            time.sleep(5)
            print(f'process {i} are woring')

        # "1.1 RandomForest"
        # print('####################### 当前运行的模型的RandomForest ######################')
        # RF_Model = RandomForest.RF_model(
        #                             RandomForest.input_data_process(traingdataframe,'train_test'), # 进行模型的训练和测试
        #                             RandomForest.input_data_process(traingdataframe,'prediction') # 使用全部数据的训练
        #                                 ) # 生成自己的model
        # RandomForest.Model_prediction(
        #                             RF_Model, #模型
        #                             preddate, # 日期数据
        #                             RandomForest.input_data_process(preddataframe,'prediction') # 预测的数据
        #                               )
        #
        # "1.2 PSO-RandomForest"
        # print('####################### 当前运行的模型的PSO-RandomForest #######################')
        # best_parameters = PSO_RandomForest.PSO_RF_Model(PSO_RandomForest.input_data_process(traingdataframe,'train_test'))
        # PSO_RF_Model = PSO_RandomForest.Model_prediction(
        #
        #                                             best_parameters, # 模型参数
        #                                             preddate, # 模型运行的日期
        #                                             PSO_RandomForest.input_data_process(traingdataframe,'train_test'), # 拟合的数据
        #                                             PSO_RandomForest.input_data_process(traingdataframe,'prediction'), # 最终的训练数据
        #                                             PSO_RandomForest.input_data_process(preddataframe,'prediction') # 预测的数据
        #                                                 )
        #
        # "1.2 PSO-SVM模型"
        # "数据处理部分：参数使用了标准化！"
        # train_testdata = Input_data.data_process1(traingdataframe) # 这个输出有四个参数，直接使用即可!注意这个参数的顺序！！
        # pred_data = Input_data.data_process2(preddataframe) # 输出两个参数，进行预测
        # "模型生成部分"
        # print('####################### 当前运行的模型师PSO-SVM #######################')
        # best_params = pso_svm.pso_svm_model(train_testdata) # pso-svm输出的最佳参数
        # pso_svm.best_model(best_params,preddate,train_testdata,pred_data)

        "1.3 GBDT_PSO_SVM模型"
        # trainX_new,trainy,testX_new,testy,predX_new,predy = GBDT(train_testdata,pred_data) #由于GBDT产出的特征更多，因此会出现无法拟合的情况
        "模型生成部分"
        # print('当前运行的模型是GBDT-PSO-SVM，由于样本不足，因此是无法输出结果的')
        # best_params = pso_svm.pso_svm_model(train_testdata) # pso-svm输出的最佳参数
        # new_traindata,new_preddata = [trainX_new,trainy,testX_new,testy],[predX_new,predy]
        # pso_svm.best_model(best_params,new_traindata,new_preddata) # 用最佳参数的模型 + 结合新特征的数据


        "2. 比较模型！: "
        "logit"
        # std_tag=True
        # lg = logit_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "logistic"
        # std_tag = False
        # lr = logistic_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "nb"
        # std_tag = False
        # gnb = Navbay_(data_preprocess(traingdataframe,preddataframe,std_tag))
        "SVC"
        # std_tag = True
        # svc = svc_(data_preprocess(traingdataframe,preddataframe,std_tag))

        "3. 功能性增强"
        "绘制1：calibration curve"
        # lr = LogisticRegression()
        # gnb = GaussianNB()
        # svc = NaivelyCalibratedLinearSVC(C=1.0)
        # rfc = RandomForestClassifier()
        # clflist = [(lr,'logistic',False),(gnb,'Naive Bayes',False),(svc,'SVC',True),(rfc,"Random Forest",False)]
        # Main_plot.calibration_plot(clflist,traingdataframe,preddataframe)

        "绘制2：结果的可解释性：这部分可以直接在模型中运行"

    return None

"考虑多进程工作！逐个唤起不同的模型"
"Model1: RandomForest"
def RandomForest_process(traingdataframe,preddate,preddataframe):
    print('####################### 当前运行的模型的RandomForest ######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/NonOptimize_RandomForest_res.txt', 'a+') as f:
        f.write(f'这是随机森林模型，工作于{preddate}')
    RF_Model = RandomForest.RF_model(
        RandomForest.input_data_process(traingdataframe, 'train_test'),  # 进行模型的训练和测试
        RandomForest.input_data_process(traingdataframe, 'prediction')  # 使用全部数据的训练
    )  # 生成自己的model
    RandomForest.Model_prediction(
        RF_Model,  # 模型
        preddate,  # 日期数据
        RandomForest.input_data_process(preddataframe, 'prediction')  # 预测的数据
    )
    return None

"Model2: PSO-RandomForest"
def PSO_RandomForest_process(traingdataframe,preddate,preddataframe):
    print('####################### 当前运行的模型的PSO-RandomForest #######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/optimize_RandomForest_res.txt', 'a+') as f:
        f.write(f'这是PSO优化的随机森林模型，工作于{preddate}')
    best_parameters = PSO_RandomForest.PSO_RF_Model(PSO_RandomForest.input_data_process(traingdataframe, 'train_test'))
    PSO_RF_Model = PSO_RandomForest.Model_prediction(
        best_parameters,  # 模型参数
        preddate,  # 模型运行的日期
        PSO_RandomForest.input_data_process(traingdataframe, 'train_test'),  # 拟合的数据
        PSO_RandomForest.input_data_process(traingdataframe, 'prediction'),  # 最终的训练数据
        PSO_RandomForest.input_data_process(preddataframe, 'prediction')  # 预测的数据
    )
    return None


"Model3： PSO-SVM"
def PSO_SVM_process(traingdataframe,preddate,preddataframe):
    "数据处理部分：参数使用了标准化！"
    train_testdata = Input_data.data_process1(traingdataframe)  # 这个输出有四个参数，直接使用即可!注意这个参数的顺序！！
    pred_data = Input_data.data_process2(preddataframe)  # 输出两个参数，进行预测
    "模型生成部分"
    print('####################### 当前运行的模型师PSO-SVM #######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_SVM_res.txt', 'a+') as f:
        f.write(f'这是PSO-SVM模型工作于{preddate}')
    best_params = pso_svm.pso_svm_model(train_testdata)  # pso-svm输出的最佳参数
    pso_svm.best_model(best_params, preddate, train_testdata, pred_data)
    return None


"Model4: PSO-RFE-SVM"
def PSO_RFE_SVM_process(traingdataframe,preddate,preddataframe):
    train_testdata = Input_data.data_process1(traingdataframe) # 输出需要训练模型的基础数据集
    "模型生成部分"
    print('###################### 当前运行的模型是PSO-RFE-SVM ######################')
    with open('/home/haozhic2/ListedCompany_risk/Results_output/Source_file/PSO_RFE_SVM_res.txt','a+') as f:
        f.write(f'这是PSO-RFE-SVM模型工作于{preddate}')
    best_params = RFE_PSO_SVM.pso_svm_model(train_testdata)
    RFE_PSO_SVM.best_model(best_params, preddate, traingdataframe, preddataframe)

'''说明
1）SVC方法存在一个问题：参数选择的时候模型C参数必须等于1
'''



if __name__ == '__main__':
    main()