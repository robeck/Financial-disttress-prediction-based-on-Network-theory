'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现将数据输入，得到DEA的输出

注意：
1. 需要处理不同时间
'''
import pandas as pd
import numpy as np
from Model.DEA import DEA


'''对DEA方法的调用
目前的方案考虑：将程序编写成更加可扩展的形式
input
    1）date
    2）file
    3）stocks
output
    1）DEA的结果

'''
def dea_cal(files,stockdata,date,dimtags,dea_method):
    # 默认参数
    df_financial,df_growth,df_innov,df_operation,df_output = pd.read_csv(files[0]),pd.read_csv(files[1]),pd.read_csv(files[2]),\
                                                             pd.read_csv(files[3]),pd.read_csv(files[4])
    research_stock = stockdata[['Stkcd']] # 研究的ST，非ST公司
    
    "1. 数据处理，合并"
    df_financial_date = df_financial[df_financial['Date']==date].set_index('Stkcd')
    df_growth_date = df_growth[df_growth['Date']==date].set_index('Stkcd')
    df_innov_date = df_innov[df_innov['Date']==date].set_index('Stkcd')
    df_operation_date = df_operation[df_operation['Date']==date].set_index('Stkcd')
    df_output_date = df_output[df_output['Date']==date].set_index('Stkcd')
    mergedata = pd.concat([df_financial_date,df_growth_date,df_innov_date,df_operation_date,df_output_date],axis=1,join='inner') # 横向合并了数据！
    mergedata = mergedata.fillna(0).drop('Date',axis=1).reset_index() # 从新将index'Stkcd'-> 转到column中
    "mergedata中数据太大的问题要处理一下！即缩小量纲！"
    mergedata = mergedata.copy()
    mergedata['totalasset'] = [asset/1000000 for asset in mergedata.totalasset.tolist()]
    mergedata['R&Dcost'] = [rdcost/100 for rdcost in mergedata['R&Dcost'].tolist()]
    mergedata['operationcost'] = [np.abs(opcost/10000) for opcost in mergedata.operationcost.tolist() ]
    mergedata['revenue'] = [revenue/10000 for revenue in mergedata.revenue.tolist()]


    "1.1 如果每类输入指标 -> 分别对应输出指标呢？"
    df_financial_output = mergedata[['Stkcd','currentflow','currencyrate','debetasset','mainproportions','totalasset','revenue','ROA']]
    df_growth_output = mergedata[['Stkcd','fixinvesgrowth', 'operincomegrowth', 'operprofitgrowth','revenue', 'ROA']]
    df_innov_output = mergedata[['Stkcd','Innovperson', 'R&Dcost', 'R&Dgrowth','revenue', 'ROA', 'ROE']]
    df_operation_output = mergedata[['Stkcd','fixturn', 'operationturn','operationcost','revenue', 'ROA']]
    "1.2 如果我们只选择部分数据进行运算，即非完整数据，考虑测试"
    test_data = mergedata[['Stkcd','fixturn','R&Dcost','debetasset','operationcost','currentflow','revenue','ROA','ROE']]

    tagsdict = {'financial':df_financial_output,'growth':df_growth_output,'innovation':df_innov_output,'operation':df_operation_output,'total':test_data} # 构建一个tag：data的dictionary

    "2. 制作DEA处理需要的数据"
    "将数据划定到研究范围内！"
    research_data = tagsdict.get(dimtags) # 根据我们输入的标签，匹配tagdict，选择除进行研究的数据！
    research_data = pd.merge(research_stock,research_data,how='inner',on='Stkcd').set_index('Stkcd') #这里对merge中（a，b） b进行调整
    X = research_data.iloc[:,:-2]
    Y = research_data.iloc[:,-2:]
    # print(X.head())
    # print(Y.head())

    "3. 运行"
    dea = DEA(data=research_data,DMUs_Name=research_data.index,X=X,Y=Y) # 创建实例
    res = dea.analysis(dea_method,None,None) # 运行CRS结果
    "增加日期列" # 可以不用，未必需要
    # res['Date'] = date

    return res



def DEA_main(data,stockdate,tag,dea_method):
    '''
    :param data: ST和非ST公司统计
    :param date: 研究的日期
    :return: DEA的结果
    '''
    # 文件就是全部的DEA需要的数据，根据结果可以进行调整和调换
    financial_file = '~/ListedCompany_risk/Data/DEAdata/financial_final_data.csv'
    growth_file = '~/ListedCompany_risk/Data/DEAdata/growth_final_data.csv'
    innovation_file = '~/ListedCompany_risk/Data/DEAdata/innovation_final_data.csv'
    operation_file = '~/ListedCompany_risk/Data/DEAdata/operation_final_data.csv'
    output_file = '~/ListedCompany_risk/Data/DEAdata/output_final_data.csv'

    filelist = [financial_file,growth_file,innovation_file,operation_file,output_file]
    res = dea_cal(filelist,data,stockdate,tag,dea_method)

    return res



if __name__ == '__main__':
    # 测试数据
    test_data = pd.DataFrame({'Stkcd':[600076,600082,600032],'ST_value':[1,0,0]})
    print(test_data)
    # 参数
    tag = 'financial' # growth, innovation,operation,total
    method = 'CRS'

    DEA_main(test_data,2015,tag,method)


