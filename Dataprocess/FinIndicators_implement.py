'''
@Author: haozhi chen
@Date: 2022-09
@Target: 作为一个读取金融指标数据的转接程序

'''
import pandas as pd

def indicators_main(researchdata,date):
    # 初始化一些参数
    date = date
    research_stocks = list(set(researchdata.Stkcd.tolist())) # 规避重复数据
    "数据读取"
    indicators_df = pd.read_csv('~/ListedCompany_risk/Data/Financialdata/financial_res_data.csv')

    "1. 数据的截取 (时间，研究对象)"
    indicators_df_date = indicators_df[indicators_df['Date']==date] # 正确的研究日期
    indicators_df_date = indicators_df_date.reset_index().drop('index',axis=1) # 重置索引
    rows = [row for i,row in enumerate(indicators_df_date.index) if indicators_df_date.iat[i,0] in research_stocks] # 研究对象股票行
    indicators_df_date_target = indicators_df_date.iloc[rows,:] # 正确日期，研究对象公司的 数据

    "2. 返回数据"
    indicators_df_date_target = indicators_df_date_target.drop('Date',axis=1)
    indicators_df_date_target = indicators_df_date_target.reset_index().drop('index',axis=1)


    return indicators_df_date_target


if __name__ == '__main__':
    pass
    df = pd.read_csv('~/ListedCompany_risk/Data/Financialdata/financial_res_data.csv')
    print(df)
    print(df.columns)

