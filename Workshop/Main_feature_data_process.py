''':
@Author: haozhi chen
@Date: 2022-09
@Target: 对network得到的数据，dea得到的数据，金融数据进行一个汇总

注意：
2023-02 版本：
(1)考虑不使用DEA数据了，因为效率指标只产生一个，影响极为有限。并且所生成的指标也存在一些偏差，因此考虑不用
(2)考虑调整一些金融指标，并不使用全部的金融指标，只使用一些真正关键的！
'''
import pandas as pd
import numpy as np
from Dataprocess import DEA_implement,Risk_stock_process
from Dataprocess import FinIndicators_implement
from NetworkConstruction.TopoImplement import Implementation
from multiprocessing import Process
import logging
import os

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

"网络拓扑结果抽去"
def Network_res_input(date,logger,impletag,filetag):
    logger.info(f'The network working on date {date}')
    if filetag == 'new':
        filename = f'/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_{date}_{filetag}.txt' # 当期年份的波动率溢出网络
    else:
        filename = f'/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_{date}.txt'  # 当期年份的波动率溢出网络
    res = Implementation.single_date_implement(filename,impletag)
    logger.info(f'The {date} network working end')
    return res


"DEA结果抽取"
def DEA_res_input(stockdata,date,tags,logger,dea_method):
    res = pd.DataFrame()
    try:
        logger.info(f'The DEA working corrected {tags} on date {date}')
        deadata = DEA_implement.DEA_main(stockdata,date,tags,dea_method)
        res = pd.DataFrame(deadata)
    except (AttributeError):
        logger.warning(f'working error on {date}, {tags}')
        print('The attribute eerors are invoked')

    return res

"金融数据抽取"
def financial_data_input(stockdata,logger,date):
    logger.info(f'The financial data working on date {date}')
    res = FinIndicators_implement.indicators_main(stockdata,date)
    logger.info(f'The {date} financial data working end')
    return res

"填充列中nan数据为均值的方法"
def fill_nan_with_mean(data):
    output = data.copy()  # 复制创建一个新dataframe
    "先判断"
    bool_data = data.isna()
    booldetermine = bool_data.any().any()
    print(f"-----------------------数据nan值检验为 {booldetermine}----------------------")
    "列均值填充过程"
    if booldetermine:
        #########################################################################
        output.fillna(output.mean(),inplace=True) # 更新后替换原数据
        ############################################################################
    else:
        pass
    return output

"汇总所有数据"
''':arg
'''
def interge_all(stocktagres,networkres,deares,financialres):
        # 必须说明，链接的索引到底是index 还是 columns
    mergedata = pd.merge(stocktagres,pd.merge(networkres,pd.merge(deares,financialres,
                                                                  how='inner',on='Stkcd'),
                                              how='inner',on='Stkcd'),
                         how='inner',on='Stkcd')
    return mergedata

"汇总两组数据！"
'''
这个主要是对基础的一些数据进行简单的测试！
'''
def intege_two(stockatgres,other):
    mergedata = pd.merge(other,stockatgres,how='inner',on='Stkcd') # 调转一下数据，让标签放置在最后
    return mergedata

"汇总三组数据，没有dea的"
def intege_three(stockatgres,networkres,financialres):
    mergedata = pd.merge(stockatgres,pd.merge(networkres,financialres,how='inner',on='Stkcd'),
                         how='inner',on='Stkcd')
    return mergedata


########################################################################################################################

"多进程的安排"
''':说明
（1）2023-02增加logger记录器
（2）对nan数据进行填补！列均值方案
'''
def main_data_multiple_process(date,deatags,dea_method,networkcuttag):
    filetag = 'new'  # 'new' or 'None' # New 表示读取新构建的样本网络数据，None表示旧的

    print(f'-----------------The current intergrate research data on {date}--------------------')
    "读取有，无风险标识的股票"
    df_ST, df_nonST, df_stocktags = Risk_stock_process.stocklist(date)

    stocks = df_stocktags.Stkcd.tolist()
    stocks.sort()
    print(stocks)
    print(len(stocks))

    "1) 生成DEA的results，这里面包含了很多测试成分，需要tag"
    # deares = DEA_res_input(df_stocktags, date, deatags, logger, dea_method)
    # print(deares.describe())
    # deastocklist = deares.Stkcd.tolist()

    "2) Network中的拓扑结构数据提取"
    topolres = Network_res_input(date, logger,networkcuttag,filetag)  # 网络是用df_total中的stock构建的
    print(topolres.describe())
    topostocklist = topolres.Stkcd.tolist()

    "3) 金融预测的financial数据提取"
    financialres = financial_data_input(df_stocktags,logger, date)
    print(financialres.describe())
    finalstocklist = financialres.Stkcd.tolist()

    # print(f'{date}DEA数据股票list长度为：{len(deastocklist)}')
    # deastocklist.sort()
    # print(deastocklist)
    print(f'{date}topo数据股票list长度为：{len(topostocklist)}')
    topostocklist.sort()
    print(topostocklist)
    print(f'{date}final数据股票list长度为：{len(finalstocklist)}')
    finalstocklist.sort()
    print(finalstocklist)

    "final 数据的合并"
    # temp_integredata = interge_all(df_stocktags, topolres, deares, financialres)  # 返回一个dataframe
    temp_integredata = intege_three(df_stocktags, topolres, financialres) # 返回一个没dea的数据，避免dea总错误袭扰
    "列均值填充missing value过程"#####################################
    temp_integredata = fill_nan_with_mean(temp_integredata)
    ###################################################
    temp_integredata['Date'] = date  # 这个必不可少，因为df_stocktags中没有date日期
    # temp_integredata.to_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}.csv',index=False) # 原始网络的数据
    temp_integredata.to_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}_{filetag}.csv', # 新网络的数据
                            index=False)
    print(f'融合后的股票list长度为：{len(temp_integredata.Stkcd.tolist())}')

    return None

"*补充程序：金融数据的重构!:"
''':说明
我们的主要思路是：
（1）在Input data process中重新构建findata的数据集合，
（2）在这里调用integrated_data_date, 剔除其中的financialres部分，
（3）integrated_data_date 和新的findata数据进行合并
'''
def repalce_financail_part(start_date,end_date):
    ''':param
    日期范围
    '''
    for date in range(start_date,end_date):
        "1) 读取temp integrated数据"
        temp_integredata = pd.read_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}.csv') #读取数据
        temp_integredata = temp_integredata.iloc[:,0:8] # 只保留网络指标
        "2) 金融预测的financial数据提取"
        df_ST, df_nonST, df_stocktags = Risk_stock_process.stocklist(date)
        financialres = financial_data_input(df_stocktags, logger, date)
        print(financialres.describe())
        "3) 合并数据"
        final_integredata = intege_two(temp_integredata,financialres)
        final_integredata = fill_nan_with_mean(final_integredata)
        print(final_integredata.shape)
        print(final_integredata.head())
        final_integredata['Date'] = date  # 这个必不可少，因为df_stocktags中没有date日期
        final_integredata.to_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}.csv',index=False)

    return None

"唤起多线程操作"
''':说明
（1）deatags ： 标记dea计算涉及的数据
（2）dea method ：标记dea的方法
（3）networkcutting ：标记网络缩减的方法
'''
def multiprocess_call():
    # 多组时间范围！
    # start_date, end_date = 2015, 2022
    start_date, end_date = 2020,2021
    # start_date, end_date = 2011, 2022

    # deatags = ['financial','innovation','growth','operation','total'] # 多组结果测试用
    deatags = 'total'  # DEA计算哪些数据的标签
    # dea_method = 'CRS'  # DEA方法的标签
    dea_method = 'VRS' # VRS方法，根据一些研究VRS相对较好。暂时SBM模型的结果不如人意料
    networkcuttag = 'Cut_zero'  # 网络缩减的标签

    "多进程进行，加快计算"
    for date in range(start_date, end_date): # 从2015-2022年，每一年
        args = date,deatags,dea_method,networkcuttag # 每一年的参数
        process = Process(target=main_data_multiple_process, args=args)
        process.start()
        print(f'processes {process.name} are working')

    return None

"多进程运行完成后，执行该步骤，汇总数据"
''':说明
1）在当前程序中调用，生成的数据进行保存
2）通过Main_function进行调用，其中的通过参数调控
'''
def multiprocess_merge(start_date, end_date,tags,nettags):
    start_date, end_date = start_date, end_date #

    "每个进程结束输出到文本的数据读取出来，进行合并"
    integrated_data = pd.DataFrame()  # 存储汇总好的数据
    for date in range(start_date, end_date):
        if nettags == 'new':
            temp_data = pd.read_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}_{nettags}.csv')
        else:
            temp_data = pd.read_csv(
                f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{date}.csv')

        if 'Date' not in temp_data.columns:
            temp_data['Date'] = date
        else:
            pass
        print(f'The intergrated data date is {date}, the data shape is {temp_data.shape}')
        # print(temp_data.head())
        "不同日期下的 final数据 的纵向合并"
        integrated_data = pd.concat([integrated_data, temp_data], axis=0)

    "数据需要清理nan数据，那么有两种方案：1）fill，2）drop"
    print(integrated_data.columns)
    "数据的填补，删除，列位置的更换"
    # integrated_data_clean = integrated_data.fillna(0) # 填补nan数据为0，这里当然有待商榷
    # integrated_data_clean = integrated_data.dropna()  # 删除nan存在的行数据
    if tags == 'full': #这里调整剔除DEA数据，此外，我们还需要对金融指标进行调整； network indicators position 6,因为要剔除Stkcd
        integrated_data_clean = integrated_data[['Stkcd', 'total_connectedness', 'connectivity',
                                                 'closeness_centrality', 'betweenness_centrality', 'degree_centrality',
                                                 'pagerank', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio',
                                                 'Operating_income_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Fixed_asset_invest_growth_ratio', 'Net_cash_flow_growth_rate',
                                                 'Assets_to_equity_ratio', 'Long_debt_to_long_capital_ratio',
                                                 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
            # 'Ebitda_to_operatingincome_ratio','Expense_ratio'
            # 'Turnover_account_receivable','Total_asset_turnover_rate'
    elif tags == 'justed_financial': #这里调整剔除DEA数据，此外，我们还需要对金融指标进行调整
        integrated_data_clean = integrated_data[['Stkcd', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio',
                                                 'Operating_income_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Fixed_asset_invest_growth_ratio', 'Net_cash_flow_growth_rate',
                                                 'Assets_to_equity_ratio', 'Long_debt_to_long_capital_ratio',
                                                 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
    elif tags == 'adjusted_network': # network indicators position 4,因为要剔除Stkcd
        integrated_data_clean = integrated_data[['Stkcd',
                                                 'closeness_centrality', 'betweenness_centrality', 'degree_centrality',
                                                 'pagerank', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio',
                                                 'Operating_income_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Fixed_asset_invest_growth_ratio', 'Net_cash_flow_growth_rate',
                                                 'Assets_to_equity_ratio', 'Long_debt_to_long_capital_ratio',
                                                 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
    elif tags == 'adjusted_financial':# network indicators position 6,因为要剔除Stkcd
        integrated_data_clean = integrated_data[['Stkcd', 'total_connectedness', 'connectivity',
                                                 'closeness_centrality', 'betweenness_centrality', 'degree_centrality',
                                                 'pagerank', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Net_cash_flow_growth_rate', 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
    elif tags == 'justed_financial':
        integrated_data_clean = integrated_data[['Stkcd', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Net_cash_flow_growth_rate', 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
    elif tags == 'adjusted_network_financial': # network indicators position 4
        integrated_data_clean = integrated_data[['Stkcd', 'closeness_centrality',
                                                 'betweenness_centrality', 'degree_centrality',
                                                 'pagerank', 'Return_on_equity',
                                                 'Return_on_assets', 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate',
                                                 'Net_profit_growth_rate', 'Liquidity_ratio', 'Quick_ratio',
                                                 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Debt_to_assets_ratio', 'Turnover_account_receivable',
                                                 'Total_asset_turnover_rate',
                                                 'Operating_profit_ratio', 'Main_business_ratio', 'Invest_income_ratio',
                                                 'Non_operating_income_ratio', 'Currency_growth_rate',
                                                 'Net_cash_flow_growth_rate', 'Non_current_asset_to_equity_ratio',
                                                 'Tagible_assets_to_liabilities_ratio', 'Net_debt_to_equity_ratio',
                                                 'Current_asset_turnover_ratio', 'ST_value', 'Date']]  # 为了让ST——value
    else:
        integrated_data_clean = integrated_data[['Stkcd', 'total_connectedness', 'connectivity',
                                                 'closeness_centrality', 'betweenness_centrality', 'degree_centrality',
                                                 'pagerank',
                                                 'effeciency', # 有效率指标！
                                                 'Return_on_equity', 'Return_on_assets',
                                                 'Ebitda_to_operatingincome_ratio', 'Expense_ratio',
                                                 'Net_asset_growth_rate',
                                                 'Operating_income_growth_rate', 'Net_profit_growth_rate',
                                                 'Liquidity_ratio', 'Quick_ratio', 'Cash_to_current_ratio',
                                                 'Cash_debt_ratio', 'Debt_to_assets_ratio',
                                                 'Turnover_account_receivable', 'Total_asset_turnover_rate', 'ST_value',
                                                 'Date']]  # 为了让ST——value
    print(f'输出的整合数据的样例：{integrated_data_clean.head()}')
    print(f'输出的整合数据的shape形状：{integrated_data.shape}')
    integrated_data_clean = integrated_data_clean.sort_values(['Date','Stkcd'])
    print(integrated_data_clean.head())
    "存储到文件夹内"
    integrated_data_clean.to_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_{tags}.csv',
                                 index=False)  # 保存到文件夹内

    return integrated_data_clean


# 引入ks检验，mannwhitneyu检验
from scipy import stats
"顺手进行一下KS检验，目前看所有金融指标都是符合正太分布要求的，因此不用剔除"
def normalization_distribution_test(start_date, end_date):
    # 读取数据
    data = pd.read_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_adjusted_network.csv')
    print(data)
    # 数据清洗
    data_indicators = data.iloc[:,1:-2] # 提出stkcd，st-value，date
    # 基本展示
    print(data_indicators.head())
    print(data_indicators.shape)
    "全时间的检验"
    for column in data_indicators.columns:
        data_col = data.loc[:,column]
        kstest = stats.kstest(data_col, 'norm')
        print(f'当前是全时间内，{column}的ks检验')
        print(kstest)
        print('############################################')
    "分时间段的检验"
    # 数据清洗，需要保留时间
    # data_indicators = data.drop(columns=['Stkcd','ST_value'])
    # windowsize = 1
    # dateinterval = 3
    # for date in range(start_date, end_date - dateinterval, windowsize):  # 结尾位置 -dateinterval，实现的是最后一起3年直接预测
    #     start_date, middle_date, end_date = date, date + dateinterval - windowsize, date + dateinterval # 计算要截断的时间
    #     data_indicators_date = data_indicators[(data_indicators['Date']>=start_date) & (data_indicators['Date'] <= middle_date)] # 截断时间范围内的数据
    #     data_indicators_date = data_indicators_date.drop(columns=['Date'])
    #     for column in data_indicators_date.columns:
    #         data_col = data.loc[:, column]
    #         kstest = stats.kstest(data_col, 'norm')
    #         print(f'在{start_date}-{middle_date}内，{column}的ks检验')
    #         print(kstest)
    #         print('############################################')
    # kstest = stats.kstest(x,'norm') # 原始假设是服从正态分布，那么kstest res结果中 pvalue 越小说明不能拒绝正太分布的假设
    # print(kstest)
    return None

"进行Mann-whitney检验"
def mean_comparation():
    data = pd.read_csv(f'~/ListedCompany_risk/Data/Outputdata/Integratedata/integrated_data_adjusted_network.csv')
    data = data.sort_values(['Stkcd','Date'])
    print(data.head())
    data_st = data[data['ST_value']==1].iloc[:,1:-2]
    data_nonst = data[data['ST_value']==0].iloc[:,1:-2]
    newdata = data.iloc[:,1:-2]
    print(data_st.head())

    for column in newdata.columns:
        x = data_st.loc[:,column]
        y = data_nonst.loc[:,column]
        res = stats.mannwhitneyu(x, y)
        print(res)

    return None


if __name__ == '__main__':
    "单进程运行"
    # main_data_process() # 单进程计算

    ''':说明
    下面的两个程序不能一次性执行，必须先执行第一个，再执行第二个
    '''
    # tags = 'full' # 使用标记欠缺数据，做为最终数据标签的
    tags = 'adjusted_network'  # 筛选了部分网络指标如：只有Centrality
    # tags = 'adjusted_financial'  # 筛选了部分金融指标
    # tags = 'justed_financial'  # 只使用筛选了部分的金融指标
    # tags = 'adjusted_network_financial'  # 使用调整了网络，金融的指标
    "1. 多进程运行，先执行:单独执行"
    # 这里要说明一下，我们有原始网络，新网络两种，必须明确使用的网络！
    # multiprocess_call() # 多进程计算,计算完成后才能执行下一步 需要非常久的时间！
    "2. 后执行：单独执行"
    net_tags = 'new' # 新数据下的网络
    start_date, end_date = 2011,2022
    # repalce_financail_part(start_date,end_date) # 替换一下金融指标数据,一般可以不用的！
    data = multiprocess_merge(start_date, end_date,tags,net_tags) # 汇总多进程的结果(只需要在这里将不必要的数据排除在外就行了，原始数据该运行就运行)
    data_description = data.describe()
    data_description.to_csv('~/ListedCompany_risk/Data/Outputdata/Statistic_description/indicators_description.csv')

    "3. ks正太分布检验"
    # normalization_distribution_test(start_date, end_date)
    "4. mannwhitney检验"
    # mean_comparation()