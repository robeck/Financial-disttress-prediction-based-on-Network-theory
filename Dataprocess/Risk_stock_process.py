'''
@Author: haozhi chen
@Date: 2022-09
@Target: 专门对风险示警的公司进行读取


方案说明：
1) 读取表示ST，non-ST的数据
2）读取市值的数据（市值为nan的即不存在~）

思考：
1. 数据的选取仍然是个疑惑，如果我们选择2倍，3倍存在如下问题：
   （1）随机选择的合理性，这个模型出来的结果就不一定合适
2. 如果选择全部的数据，存在如下问题：
    （1）全部数据构建网络的时间简直就是指数倍增长
'''

import pandas as pd
import numpy as np
import random



def stocklist(date):
    random.seed(100)  # 随机一定要设定种子！不然每次都不一致！ 随机种子必须在stocklist内部，否则会出错！
    # 默认参数
    stocklist = []
    df_st,df_nonst = pd.DataFrame(),pd.DataFrame()

    "1. 读取数据"
    market_value_file = '~/ListedCompany_risk/Data/StockMarketValue.csv' #市值
    stockrisk_file = '~/ListedCompany_risk/Data/StockRiskWarning_processed.csv' #风险标签
    stockindustry_file = '~/ListedCompany_risk/Data/StockIndustry_processed.csv' # 公司所属行业标签
    df_value = pd.read_csv(market_value_file)
    df_stockrisk = pd.read_csv(stockrisk_file)
    df_stockind = pd.read_csv(stockindustry_file)
    
    "2. 可以根据输入的date截取数据"
    df_value_date = df_value[df_value['Date']==date].drop('Date',axis=1)
    df_stockrisk_date = df_stockrisk[df_stockrisk['Date']==date].drop('Date',axis=1)
    df_stockind_date = df_stockind[df_stockind['Date']==date].drop('Date',axis=1)

    "3. 设置筛选市值的要求"
    df_value_date.dropna() #剔除nan数据
    df_value_date = df_value_date[df_value_date['market_value']<1000]
    # print(df_value_date[df_value_date['Stkcd']==600817]) # 测试某一个公司
    "初探不同时期ST公司,Non-ST公司"
    df_stockrisk_value_date = pd.merge(df_stockrisk_date,
                                       pd.merge(df_value_date,df_stockind_date,how='inner',on='Stkcd'),
                                                        how='inner',on='Stkcd') # 一次性全部合并数据，合并市值，ST标签，行业
    df_stockrisk_value_date_ST = df_stockrisk_value_date[df_stockrisk_value_date['ST_value']==1]
    df_stockrisk_value_date_nonST = df_stockrisk_value_date[df_stockrisk_value_date['ST_value']==0]
    "市值范围进一步精确"
    '''
    这里的重点在于：我们要从non-st公司中选出一些作为有效的学习样本。因此我们提出了一些想法
    1）限制市值，市值范围相近
    2）行业分布一致，比较重要
    # 3) 我们必须看到当前这些公司在过去3年都是存在的，（实现方法：市值非Nan） # 
    '''
    marketvalue = df_stockrisk_value_date_ST.market_value.tolist()
    max = np.ceil(np.max(marketvalue)) # 敲定最大值
    min = np.floor(np.min(marketvalue)) # 敲定最小值
    df_range_stockrisk_value_date_nonST = df_stockrisk_value_date_nonST[(df_stockrisk_value_date_nonST['market_value']>=min) &
                                                        (df_stockrisk_value_date_nonST['market_value']<=max)] #非风险公司筛选
    df_stockrisk_value_date_ST = df_stockrisk_value_date_ST.reset_index().drop('index',axis=1) # 重置index
    df_range_stockrisk_value_date_nonST = df_range_stockrisk_value_date_nonST.reset_index().drop('index',axis=1) #重置index！！
    '''行业处理
    1）提取ST公司的行业分布
    2）寻找两倍数量的公司
    3）针对行业内的市值限制！（1）上下限范围，（2）上下限范围内幅度比例 20% # 2023-02 调整
    '''
    Indcount_ST = df_stockrisk_value_date_ST['Industry'].value_counts() # ST公司中行业分布情况
    Indcount_nonST =df_range_stockrisk_value_date_nonST['Industry'].value_counts() # Non-ST公司行业分布
    Ind_search_dic = {} #存储备份要选择的行业，以及其中公司数量
    choice_nonst = [] # 选择的非ST公司汇总
    for i,v  in enumerate(Indcount_ST):
        name = Indcount_ST.index[i] # 行业名称
        "针对行业的上市Non-ST公司设置要求:市值在行业内选择范围为ST公司上下限，可浮动" # 2023-02调整
        df_industry_stockrisk_value_date_ST = df_stockrisk_value_date_ST[df_stockrisk_value_date_ST['Industry']==name] #ST中当前行业的公司
        max_inds,min_inds = np.ceil(np.max(df_industry_stockrisk_value_date_ST.market_value.tolist())),\
                            np.floor(np.min(df_industry_stockrisk_value_date_ST.market_value.tolist())) #当前行业ST公司的市值上下限
        df_inds_range_stockrisk_value_date_nonST = df_range_stockrisk_value_date_nonST[df_range_stockrisk_value_date_nonST['Industry']==name] # 当前行业内非ST公司
        df_inds_rangeinds_range_stockrisk_value_date_nonST = df_inds_range_stockrisk_value_date_nonST[(df_inds_range_stockrisk_value_date_nonST['market_value']>=(min_inds * 0.8)) &
                                                                                            (df_inds_range_stockrisk_value_date_nonST['market_value']<=(max_inds * 1.2))] #当前行业内非ST公司符合上下限要求的
        "进行选择！"
        value = v # 公司数量
        choicevalue = value*2 # 要选择的数量
        Ind_search_dic[name] = choicevalue # 存一个dic，用于备份
        ind_stock_list = df_inds_rangeinds_range_stockrisk_value_date_nonST.Stkcd.tolist() # 符合要求的non-st公司list
        choicelist = random.sample(ind_stock_list,choicevalue) # 随机选择两倍！
        choice_nonst = choice_nonst + choicelist
    choice_nonst = list(set(choice_nonst)) #清查一下重复数据

    "根据选择好的non-ST公司，从dataframe中提取df"
    row = [row for i,row in enumerate(df_range_stockrisk_value_date_nonST.index) if df_range_stockrisk_value_date_nonST.iat[i,0] in choice_nonst] # 抽取这些公司所在的列
    df_choice = df_range_stockrisk_value_date_nonST.iloc[row,:] # 形成需要的df
    df_choice = df_choice.reset_index().drop('index',axis=1)  # 重置索引！

    "Final: 输出"
    df_st = df_stockrisk_value_date_ST
    df_nonst = df_choice
    df_total = pd.concat([df_st,df_nonst],axis=0).reset_index().drop(['index','market_value','Industry'],axis=1) # 全部的公司

    df_stock_list=df_total.Stkcd.tolist()
    df_stock_list.sort()
    # print(df_stock_list)

    return df_st,df_nonst,df_total #返回选择好的ST公司，非ST公司，以及全部公司



if __name__ == '__main__':

    for date in range(2011,2022):
        print(f'The working date is {date}')
        df1,df2,df3 = stocklist(date)

        print(df1.shape)
        print(df2.shape)
        print(df3.shape)
        # stocks = df3.Stkcd.tolist()
        # stocks.sort()x
        # print(stocks)

    # market_value_file = '~/ListedCompany_risk/Data/StockMarketValue.csv'  # 市值
    # data = pd.read_csv(market_value_file)
    # data = data[data['Date']==2012]
    # print(data)


