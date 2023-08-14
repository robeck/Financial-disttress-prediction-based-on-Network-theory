'''
@Authoe: haozhi chen
@Date: 2022/09
@Target: 处理股票市场special treat的数据

'''
import pandas as pd
import numpy as np
import random

''': 将Wind结构的数据转换成一般使用的习惯性结构

'''
def data_process():
    # 上市公司ST数据处理
    ST_data_filename = '~/ListedCompany_risk/Data/StockRiskWarning_vertical.csv'
    df = pd.read_csv(ST_data_filename).set_index('Stkcd')

    "index columns pre-process"
    df.index = [int(x[0:6]) for x in df.index]
    df.columns = [x[0:4] for x in df.columns]

    "否 -> 0, 是 -> 1"
    empty_dataframe = pd.DataFrame()
    for column in df.columns:
        df[column] = [1 if x=='是' else 0 for x in df[column]]
        temp_df = df[[column]]
        date = temp_df.columns.tolist()[0]
        temp_df['Date'] = date
        temp_df = temp_df.reset_index().rename(columns={column:'ST_value','index':'Stkcd'})
        temp_df = temp_df[['Stkcd','Date','ST_value']]
        empty_dataframe = pd.concat((empty_dataframe,temp_df),axis=0)
    finaldata = empty_dataframe.reset_index().drop('index',axis=1)

    "data -> 存储处理好的文件"
    finaldata.to_csv('~/ListedCompany_risk/Data/StockRiskWarning_processed.csv',index=False)
    return None

'''上市公司市值数据处理

'''
def data_process2():
    # 上市公司市值数据处理
    market_value_file = '~/ListedCompany_risk/Data/Listedcompany_marketvalue.csv'
    df_market = pd.read_csv(market_value_file)

    df_market['Stkcd'] = [int(x[0:6]) for x in df_market.Stkcd.tolist()]
    df_market = df_market.set_index('Stkcd')

    empty_dataframe = pd.DataFrame()
    for column in df_market.columns:
        df_column = df_market[[column]]
        df_column = df_column.rename(columns={column:'market_value'})
        df_column['Date'] = column[0:4]
        df_column = df_column.reset_index().rename(columns={'index':'Stkcd'})
        empty_dataframe = pd.concat((empty_dataframe, df_column), axis=0)

    finaldata = empty_dataframe.reset_index().drop('index', axis=1)
    print(finaldata)

    finaldata.to_csv('~/ListedCompany_risk/Data/StockMarketValue.csv',index=False)
    return None

'''股票周度交易数据合并
1）之前用CMD合并出现了格式问题，选择python合并
input:
    1)Trd-week
    2)Trd_week1
output:
    1)merge好的数据
'''
def data_process3():
    trd1_filename = '~/ListedCompany_risk/Data/TRD_Week.csv'
    trd2_filename = '~/ListedCompany_risk/Data/TRD_Week1.csv'
    df1,df2 = pd.read_csv(trd1_filename),pd.read_csv(trd2_filename)
    df_merge = pd.concat([df1,df2],axis=0)
    df_merge.sort_values(['Stkcd','Trdwnt'],inplace=True)
    # df_merge.to_csv('~/ListedCompany_risk/Data/StockWeekly.csv')
    
    "描述统计"
    df_merge_desciption = df_merge.describe()
    df_merge_desciption.to_csv('~/ListedCompany_risk/Data/Outputdata/Statistic_description/stockweekly_description.csv')
    
    return None

'''处理上市公司所属行业
'''
def data_process4():
    filename = '~/ListedCompany_risk/Data/StockIndustry.csv'
    df = pd.read_csv(filename)

    "数据处理"
    df = df.drop('日期',axis=1).rename(columns={'Date':'Stkcd'}).set_index('Stkcd')
    df.index = [int(stock[0:6]) for stock in df.index]
    df.columns = [col[0:4] for col in df.columns]

    "纵向合并过程"
    empty_dataframe = pd.DataFrame()
    for index,year in enumerate(df.columns):
        df_year = df[year].reset_index().rename(columns={'index':'Stkcd',year:'Industry'})
        df_year['Date'] = year
        empty_dataframe = pd.concat([empty_dataframe,df_year],axis=0)
    empty_dataframe = empty_dataframe.reset_index().drop('index',axis=1)
    empty_dataframe.to_csv('~/ListedCompany_risk/Data/StockIndustry_processed.csv',index=False)
    print(empty_dataframe)

"5. 对用于预测的金融数据进行读取+处理汇总，最终保存到Financialdata文件夹下"
''':说明
1）这里对原始WIND结构的数据进行读入
2）随时可能调整一下用于预测的数据
'''
def data_process5():
    # 首先设置一些必须的参数
    indicators_dataframelist = []  # 存储全部读取到的wind上的金融数据
    csv_tail = '.csv'
    "1.文件读取"
    filelists = ['Return_on_equity', 'Return_on_assets', 'Net_asset_growth_rate', 'Operating_income_growth_rate',
                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio', 'Cash_to_current_ratio',
                 'Cash_debt_ratio','Ebitda_to_operatingincome_ratio','Expense_ratio',
                 'Debt_to_assets_ratio', 'Turnover_account_receivable', 'Total_asset_turnover_rate']

    newfilelists = ['Operating_profit_ratio','Main_business_ratio','Operating_income_ratio','Invest_income_ratio',
                    'Non_operating_income_ratio','Currency_growth_rate','Fixed_asset_invest_growth_ratio','Net_cash_flow_growth_rate',
                    'Assets_to_equity_ratio','Long_debt_to_long_capital_ratio','Non_current_asset_to_equity_ratio',
                    'Tagible_assets_to_liabilities_ratio','Net_debt_to_equity_ratio','Current_asset_turnover_ratio']

    finalfile = ['Return_on_equity', 'Return_on_assets','Net_asset_growth_rate', 'Operating_income_growth_rate',
                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio', 'Cash_to_current_ratio',
                 'Cash_debt_ratio','Ebitda_to_operatingincome_ratio','Expense_ratio',
                 'Debt_to_assets_ratio', 'Turnover_account_receivable', 'Total_asset_turnover_rate',
                 'Operating_profit_ratio','Main_business_ratio','Operating_income_ratio','Invest_income_ratio',
                    'Non_operating_income_ratio','Currency_growth_rate','Fixed_asset_invest_growth_ratio','Net_cash_flow_growth_rate',
                    'Assets_to_equity_ratio','Long_debt_to_long_capital_ratio','Non_current_asset_to_equity_ratio',
                    'Tagible_assets_to_liabilities_ratio','Net_debt_to_equity_ratio','Current_asset_turnover_ratio']
    root_path = '~/ListedCompany_risk/Data/Financialdata/'

    for i, file in enumerate(finalfile):
        print(file)
        filename = root_path + file + csv_tail # 组成文件的全部路径，绝对路径
        temp_df = pd.read_csv(filename, encoding='utf-8')
        indicators_dataframelist.append(temp_df)
        print(temp_df.shape)

    print(len(indicators_dataframelist))

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
    dflists_processed = []  # 处理后的df
    '''说明
    问题：先纵向，再横向合并，会出现索引不唯一，导致横向合并难以继续
    解决：先横向合并，在纵向合并
    '''
    "2.1 数据修饰,基本结构处理"
    for index, df in enumerate(indicators_dataframelist):  # enumerate 真的是个好东西！常用！同时给到索引，值
        df.rename(columns={'日期': 'Name', 'Date': 'Stkcd'}, inplace=True)
        df['Stkcd'] = [int(stkcd[0:6]) for stkcd in df.Stkcd.tolist()]  # stkcd 转换成 list 并且截取，integer处理
        df.drop(columns=['Name'], inplace=True)
        df.set_index(['Stkcd'], inplace=True)
        df.columns = [col[0:4] for col in df.columns]
        dflists_processed.append(df)

    i = random.randint(0, 2)  # 随机选取索引位置
    columns = dflists_processed[i].columns  # 该数据的columns: 一般就是个时间序列[2011,2012,2013，。。。2021]
    "2.2 横向合并"
    horiza_dataframe = pd.DataFrame()  # 横向合并的数据集
    for jndex, col in enumerate(columns):
        "2.2 纵向合并"
        vertic_dataframe = pd.DataFrame()  # 纵向合并的数据集
        for index, df in enumerate(dflists_processed):
            df_col = df[[col]]
            df_op = df_col.rename(columns={col: finalfile[index]})
            vertic_dataframe = pd.concat([vertic_dataframe, df_op], axis=1, join='outer')

        vertic_dataframe.insert(vertic_dataframe.shape[1], 'Date', col)  # 插入时间
        horiza_dataframe = pd.concat([horiza_dataframe, vertic_dataframe], axis=0)  # 纵向合并

    horiza_dataframe.reset_index(inplace=True)  # 将Stkcd从索引 -> 列
    horiza_dataframe.to_csv('~/ListedCompany_risk/Data/Financialdata/financial_res_data.csv', index=False)
    print(horiza_dataframe)
    "数据的描述统计"
    data_description = horiza_dataframe.describe()
    print(data_description)
    data_description.to_csv('~/ListedCompany_risk/Data/Outputdata/Statistic_description/financialdata_description.csv')
    return None


if __name__ == '__main__':
    # data_process()
    # data_process2()
    # data_process3()
    # data_process4()
    data_process5()