'''
@Author: haozhi chen
@Date: 2022-09
@Target：对需要进行dcc运算的数据进行初步处理，并且计算其结果

!!注意：我们又不去做预测，我们只需要每年构建一个网络不就行了？
1）我们是每年构建网络！
2）滚动进行预测（这里不需要考虑）
'''
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm
from multiprocessing import Process
# from NetworkConstruction import Network,RelationConstruction
from NetworkConstruction.LayerDataProcess import read_data,data_preprocess
from NetworkConstruction.NetworkConstruct import layer_construct_

''':这里是Network构建的主函数
我们撰写不同class的作用各不相同
1）这里是为了存储生成的网络
2）——networkconstruction部分则是 不考虑滚动的，单期的网络
3）——relationconstruction部分则是，根据计算的dcc统计出网络节点间的relation系数

PS：我们的撰写结构应该是非常明确的！
'''
def main(filename):
    filelists = ['~/ListedCompany_risk/Data/StockRiskWarning_processed.csv',# 包含了ST信息的上市公司11年-22年的数据
                 '~/ListedCompany_risk/Data/StockWeekly.csv',# 上市公司的全部交易数据（周度，还是日度，需要考虑）
                 '~/ListedCompany_risk/Data/StockMarketValue.csv']# 上市公司市值数据
    datalist = read_data(filelists)
    Yearlist = np.arange(2011,2022) # 生成一个2011-2022的时间list

    "循环每一年，构建每一年数据的网络"
    networklayer = []
    for i in tqdm(range(len(Yearlist))):
        start_date = Yearlist[i]
        stocklist_period,data_period = data_preprocess(datalist,start_date)
        "检查一下数据处理的输出"
        # print(data_period)
        # print(len(stocklist_period))

        "进入网络构建部分，返回的是个网络"
        ''':arg
        input: 
            1）period时期的数据
            2）period时期的股票list
            3）日期
        output：
            1）网络！
        '''
        network = layer_construct_(data_period,stocklist_period,start_date)
        networklayer.append(network) # 网路文件存储

    # 将graph全部对象的list全部存入txt文件中
    # 使用pickle dump和load函数，实际上是对json结构数据的一种扩展
    # 值得注意的是。pickle必须使用绝对路径，而不能是相对路径
    f = open(filename, 'wb')
    pickle.dump(networklayer,f,0) # 网络数据输出到txt文件中

    return None


''':这里的multiprocess不进行真实部署
'''
def run_process(arglist):
    for i in range(len(arglist)):
        args = arglist[i]
        process = Process(target=main,args=args)
        process.start()
        print('processes are working')


def invoke_process():
    yearlist = np.arange(2011,2022)
    savedfilelist = []
    for year in yearlist:
        filename = f'/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_{year}.txt'
        savedfilelist.append(filename)

    arglist = zip(savedfilelist,yearlist)
    print(arglist)



if __name__ == '__main__':

    network_savedefile = '/home/haozhic2/ListedCompany_risk/Data/Networkdata/network.txt'
    main(network_savedefile)