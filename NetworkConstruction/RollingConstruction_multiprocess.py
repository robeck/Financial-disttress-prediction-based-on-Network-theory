'''
@Author: haozhi chen
@Date: 2022-09
@Target：对需要进行dcc运算的数据进行初步处理，并且计算其结果

!!注意：我们又不去做预测，我们只需要每年构建一个网络不就行了？
1）我们是每年构建网络！
2）滚动进行预测（这里不需要考虑）

想法：2023-02 不予实施。存在网络构建复杂，ST样本难以复用的问题
有一个新的网络构建思路：使用预测期的股票标的，生成T-3期的网络。间隔一年逐步循环。
因此，我们需要重新编写一个新的网络生成逻辑，并且存储到新的txt文件中
'''
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import time
from tqdm import tqdm
from multiprocessing import Process
# from NetworkConstruction import Network,RelationConstruction
from NetworkConstruction.LayerDataProcess import read_data, data_preprocess
from NetworkConstruction.NetworkConstruct import layer_construct_

''':这里是Network构建的主函数
我们撰写不同class的作用各不相同
1）这里是为了存储生成的网络
2）——networkconstruction部分则是 不考虑滚动的，单期的网络
3）——relationconstruction部分则是，根据计算的dcc统计出网络节点间的relation系数

PS：
1) 我们的撰写结构应该是非常明确的！
2）我们在多进程结构下，不需要循环时间！！ 这点非常重要
'''
def main(filename,start_date):
    filelists = ['~/ListedCompany_risk/Data/StockRiskWarning_processed.csv',  # 包含了ST信息的上市公司11年-22年的数据
                 '~/ListedCompany_risk/Data/StockWeekly.csv',  # 上市公司的全部交易数据（周度，还是日度，需要考虑）
                 '~/ListedCompany_risk/Data/StockMarketValue.csv']  # 上市公司是指数据
    datalist = read_data(filelists)

    networklist = []
    stocklist_period, data_period = data_preprocess(datalist, start_date)
    "检查一下数据处理的输出"
    # print(data_period)
    # print(len(stocklist_period))

    network = layer_construct_(data_period, stocklist_period, start_date)
    networklist.append(network) # 这里必须用list来存储，因为只有list才能被pickle模块存储到txt文件中

    print(f'测试网络节点：{network.getNodes()}')

    # 将graph全部对象的list全部存入txt文件中
    # 使用pickle dump和load函数，实际上是对json结构数据的一种扩展
    # 值得注意的是。pickle必须使用绝对路径，而不能是相对路径
    f = open(filename, 'wb')
    pickle.dump(networklist, f, 0)  # 因为只有一个网络，那么这个网路数据输出到txt文件中

    return None


''':这里的multiprocess进行真实部署
1)配置参数
2）进行调度
'''
def run_process(arglist):
    for i in range(len(arglist)):
        args = arglist[i]
        process = Process(target=main, args=args)
        process.start()
        time.sleep(10)
        print('processes are working')

"多进程调度"
def invoke_process(tags):
    start,end = 2011,2022
    "1.多进程参数配置"
    yearlist = np.arange(start, end)
    savedfilelist = []
    for year in yearlist:
        filename = f'/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_{year}_{tags}.txt'
        savedfilelist.append(filename)

    arglist = list(zip(savedfilelist, yearlist))
    "2.调配多进程"
    run_process(arglist)



if __name__ == '__main__':
    tags = 'new' # 新数据选择标准下的实验，因此要是用新tags。不需要的话可以删除还原
    invoke_process(tags)
