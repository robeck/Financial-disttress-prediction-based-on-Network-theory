'''
@Author: haozhi chen
@Date: 2022-09
@Target: 对输入的全部DEA数据，进行汇总和处理！因为数据很多，这里需要仔细谨慎

'''
import pandas as pd
import numpy as np
import random

random.seed(100)

''':运营数据的重组
时间周期：2011-2021
1）数据读取
2）结构化处理（索引，无用列，重命名等）
3）先横向，再纵向合并
'''
def operation_data_process():
    # 首先确定数据的位置
    fixasset_filename = '~/ListedCompany_risk/Data/DEAdata/Operation_input/fix_asset_turn_rate_wind_year.csv'
    operationasset_filename = '~/ListedCompany_risk/Data/DEAdata/Operation_input/operation_asset_turn_rate_wind_year.csv'
    operationcost_filename = '~/ListedCompany_risk/Data/DEAdata/Operation_input/operation_cost_PTI_wind_year.csv'

    "1.读取数据"
    df1 = pd.read_csv(fixasset_filename,encoding='utf-8') # 读取的 “固定资产周转率”
    df2 = pd.read_csv(operationasset_filename,encoding='utf-8') # 读取 ”运营资产周转率“
    df3 = pd.read_csv(operationcost_filename,encoding='utf-8') # 读取 ”运营成本“

    print(f'读取的运营方面的数据有，固定资产周转率：{df1.shape},运营资产周转率：{df2.shape},运营成本：{df3.shape}')

    "2.数据从wind转我们习惯的多列结构"
    '''
    :input
        1. 列为日期
        2. 行为公司
        该数据结构适合于时间周期不长的数据，因此，我们应该习惯将列普遍的设置为最短的那个（公司，或者事件）
    ：output
        1. 我们循环抽取每个时间的数据
        2. 然后添加时间列
        3. 进行纵向合并即可
    '''
    dflists = [df1,df2,df3] # df形成的list
    dflists_processed = [] # 处理后的df
    operation_varnames = ['fixturn','operationturn','operationcost'] # 运营数据的变量命名

    '''说明
    问题：先纵向，再横向合并，会出现索引不唯一，导致横向合并难以继续
    解决：先横向合并，在纵向合并
    '''
    "2.1 数据修饰,基本结构处理"
    for index,df in enumerate(dflists): # enumerate 真的是个好东西！常用！同时给到索引，值
        df.rename(columns={'日期':'Name','Date':'Stkcd'},inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()] # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'],inplace=True)
        df.set_index(['Stkcd'],inplace=True)
        df.columns = [col[0:4] for col in df.columns]
        dflists_processed.append(df)

    i = random.randint(0, 2) # 随机选取索引位置
    columns = dflists_processed[i].columns # 该数据的columns
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame() # 横向合并的数据集
    for jndex,col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame() # 纵向合并的数据集
        for index,df in enumerate(dflists_processed):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col:operation_varnames[index]})
            vertic_dataframe = pd.concat([vertic_dataframe,df_op],axis=1,join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col) # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe,vertic_dataframe],axis=0) #纵向合并
    
    horiza_dataframe.reset_index(inplace=True) # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/DEAdata/operation_final_data.csv',index=False)
    print(horiza_dataframe)
    return None

'''处理创新指标
时间周期：2015-2021 （受限于数据齐全性问题）
方案如上即可！
'''
def innovation_data_process():
    # 初始设置
    datelist = [str(date) for date in range(2015,2022)] # 时间

    innovperson_filename = '~/ListedCompany_risk/Data/DEAdata/Innovation_input/Innovation_person.csv'
    RDcost_filename = '~/ListedCompany_risk/Data/DEAdata/Innovation_input/R&D_cost.csv'
    RDcostgrow_filename = '~/ListedCompany_risk/Data/DEAdata/Innovation_input/R&D_cost_growth_2014.csv'

    "1. 数据读取"
    df1 = pd.read_csv(innovperson_filename,encoding='utf-8')
    df2 = pd.read_csv(RDcost_filename,encoding='utf-8')
    df3 = pd.read_csv(RDcostgrow_filename,encoding='utf-8')

    print(f'读取的创新方面的数据，研发人员：{df1.shape},研发投入：{df2.shape},研发增速：{df3.shape}')

    "根据上面的逻辑，我们优先进行横向合并，再进行纵向"
    dflists = [df1,df2,df3]
    dflists_processed = []
    innovation_varnames = ['Innovperson','R&Dcost','R&Dgrowth']

    "2.1 数据的基本修饰和处理"
    for index,df in enumerate(dflists):
        df.rename(columns={'日期':'Name','Date':'Stkcd'},inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()]  # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'], inplace=True)
        df.set_index(['Stkcd'], inplace=True)
        df.columns = [col[0:4] for col in df.columns] # columns的时间设置为year
        "这里需要一个对不同year范围数据的规范调整"
        df = df[datelist]
        "汇总"
        dflists_processed.append(df)

    i = random.randint(0, 2)  # 随机选取索引位置
    columns = dflists_processed[i].columns  # 该数据的columns
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame()  # 横向合并的数据集
    for jndex, col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame()  # 纵向合并的数据集
        for index, df in enumerate(dflists_processed):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col: innovation_varnames[index]})
            vertic_dataframe = pd.concat([vertic_dataframe, df_op], axis=1, join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col)  # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe, vertic_dataframe], axis=0)  # 纵向合并

    horiza_dataframe.reset_index(inplace=True)  # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/DEAdata/innovation_final_data.csv', index=False)
    print(horiza_dataframe)

    return None



'''财务相关数据处理
时间范围：2011-2021
逻辑从上！
'''
def financial_data_process():
    # 默认参数设定

    currentflow_filename = '~/ListedCompany_risk/Data/DEAdata/Financial_input/Current_flow_rate.csv'
    currentratio_filename = '~/ListedCompany_risk/Data/DEAdata/Financial_input/Currrency_ratio.csv'
    debetasset_filename = '~/ListedCompany_risk/Data/DEAdata/Financial_input/Debet_asset_ratio.csv'
    proportionmain_filename = '~/ListedCompany_risk/Data/DEAdata/Financial_input/Proportion_main_business.csv'
    totalasset_filename = '~/ListedCompany_risk/Data/DEAdata/Financial_input/Total_asset.csv'

    filename_list = [currentflow_filename,currentratio_filename,debetasset_filename,proportionmain_filename,totalasset_filename]
    dflist = [] # 存储处理好的df数据的空数据集
    financial_varnames = ['currentflow','currencyrate','debetasset','mainproportions','totalasset'] # 给变量一个名字

    "1. 数据读取 + 清洁"
    '''
    1).读取数据
    2).对数据进行前面func内一致的处理
    
    特点：读取+处理+输出 搞定！
    '''
    for index,filename in enumerate(filename_list):
        df = pd.read_csv(filename)
        df.rename(columns={'日期':'Name','Date':'Stkcd'},inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()]  # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'], inplace=True)
        df.set_index(['Stkcd'], inplace=True)
        df.columns = [col[0:4] for col in df.columns] # columns的时间设置为year
        "这里需要一个对不同year范围数据的规范调整"
        "汇总"
        dflist.append(df)


    i = random.randint(0, 2)  # 随机选取索引位置
    columns = dflist[i].columns  # 该数据的columns
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame()  # 横向合并的数据集
    for jndex, col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame()  # 纵向合并的数据集
        for index, df in enumerate(dflist):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col: financial_varnames[index]})
            vertic_dataframe = pd.concat([vertic_dataframe, df_op], axis=1, join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col)  # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe, vertic_dataframe], axis=0)  # 纵向合并

    horiza_dataframe.reset_index(inplace=True)  # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/DEAdata/financial_final_data.csv', index=False)
    print(horiza_dataframe)

    return None


'''财务相关数据处理
时间范围：2011-2021
逻辑从上！
'''
def growth_data_process():

    fixassetinvesg_filename = '~/ListedCompany_risk/Data/DEAdata/Growth_input/Fix_asset_investment_growth.csv'
    operationincome_filename = '~/ListedCompany_risk/Data/DEAdata/Growth_input/Operation_incom_growth.csv'
    operationprofit_filename = '~/ListedCompany_risk/Data/DEAdata/Growth_input/Operation_profit_growth.csv'

    filename_list = [fixassetinvesg_filename, operationincome_filename, operationprofit_filename]
    dflist = []  # 存储处理好的df数据的空数据集
    growth_varnames = ['fixinvesgrowth', 'operincomegrowth', 'operprofitgrowth']  # 给变量一个名字

    "1. 数据读取 + 清洁"
    '''
    1).读取数据
    2).对数据进行前面func内一致的处理

    特点：读取+处理+输出 搞定！
    '''
    for index, filename in enumerate(filename_list):
        df = pd.read_csv(filename)
        df.rename(columns={'日期': 'Name', 'Date': 'Stkcd'}, inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()]  # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'], inplace=True)
        df.set_index(['Stkcd'], inplace=True)
        df.columns = [col[0:4] for col in df.columns]  # columns的时间设置为year
        "这里需要一个对不同year范围数据的规范调整"
        "汇总"
        dflist.append(df)

    i = random.randint(0, 2)  # 随机选取索引位置
    columns = dflist[i].columns  # 该数据的columns
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame()  # 横向合并的数据集
    for jndex, col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame()  # 纵向合并的数据集
        for index, df in enumerate(dflist):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col: growth_varnames[index]})
            vertic_dataframe = pd.concat([vertic_dataframe, df_op], axis=1, join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col)  # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe, vertic_dataframe], axis=0)  # 纵向合并

    horiza_dataframe.reset_index(inplace=True)  # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/DEAdata/growth_final_data.csv', index=False)
    print(horiza_dataframe)

    return None


def output_data_process():
    revenue_filename = '~/ListedCompany_risk/Data/DEAdata/Output/Revenue.csv'
    ROA_filename = '~/ListedCompany_risk/Data/DEAdata/Output/ROA.csv'
    ROE_filename = '~/ListedCompany_risk/Data/DEAdata/Output/ROE.csv'
    totalprofit_filename = '~/ListedCompany_risk/Data/DEAdata/Output/Total_profit.csv'

    filename_list = [revenue_filename, ROA_filename, ROE_filename, totalprofit_filename]
    dflist = []  # 存储处理好的df数据的空数据集
    output_varnames = ['revenue', 'ROA', 'ROE', 'profit']  # 给变量一个名字

    "1. 数据读取 + 清洁"
    '''
    1).读取数据
    2).对数据进行前面func内一致的处理

    特点：读取+处理+输出 搞定！
    '''
    for index, filename in enumerate(filename_list):
        df = pd.read_csv(filename)
        df.rename(columns={'日期': 'Name', 'Date': 'Stkcd'}, inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()]  # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'], inplace=True)
        df.set_index(['Stkcd'], inplace=True)
        df.columns = [col[0:4] for col in df.columns]  # columns的时间设置为year
        "这里需要一个对不同year范围数据的规范调整"
        "汇总"
        dflist.append(df)

    i = random.randint(0, 2)  # 随机选取索引位置
    columns = dflist[i].columns  # 该数据的columns
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame()  # 横向合并的数据集
    for jndex, col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame()  # 纵向合并的数据集
        for index, df in enumerate(dflist):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col: output_varnames[index]})
            vertic_dataframe = pd.concat([vertic_dataframe, df_op], axis=1, join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col)  # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe, vertic_dataframe], axis=0)  # 纵向合并

    horiza_dataframe.reset_index(inplace=True)  # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/DEAdata/output_final_data.csv', index=False)
    print(horiza_dataframe)

    return None


def data_merge_description():
    financial_file = '~/ListedCompany_risk/Data/DEAdata/financial_final_data.csv'
    growth_file = '~/ListedCompany_risk/Data/DEAdata/growth_final_data.csv'
    innovation_file = '~/ListedCompany_risk/Data/DEAdata/innovation_final_data.csv'
    operation_file = '~/ListedCompany_risk/Data/DEAdata/operation_final_data.csv'
    output_file = '~/ListedCompany_risk/Data/DEAdata/output_final_data.csv'

    files = [financial_file,growth_file,innovation_file,operation_file,output_file]
    df_financial,df_growth,df_innov,df_operation,df_output = pd.read_csv(files[0]),pd.read_csv(files[1]),pd.read_csv(files[2]),\
                                                             pd.read_csv(files[3]),pd.read_csv(files[4])
    mergedata = pd.concat([df_financial, df_growth, df_innov, df_operation, df_output], axis=1,join='inner')  # 横向合并了数据！
    mergedata_description = mergedata.describe()
    mergedata_description.to_csv('~/ListedCompany_risk/Data/Outputdata/Statistic_description/DEAdata_description.csv')


if __name__ == '__main__':
    # operation_data_process()
    # innovation_data_process()
    # financial_data_process()
    # growth_data_process()
    # output_data_process()
    data_merge_description()