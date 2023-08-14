''':
@ Author: haozhi chen
@ Date: 2023-08-10
@ Target: statistic description, exaination

'''

import pandas as pd
import numpy as np
from openpyxl import *
import scipy.stats as stats


def read_data():
    file = '~/ListedCompany_risk/Data/ten_times_results.xlsx'
    df = pd.read_excel(file)
    print(df)
    return df

# Mann-Whitney检验，假设总体其它参数相同，检验总体均值是否有差异，样本量可不同
# stats.mannwhitneyu(x, y)
def manntest(df):
    metrics = ['ACC','F1-Score','AUC','Precision','Recall']
    models = ['ADAPSO-RF','GRIDCV-RF','PSO-RF','ADABOOST-RF']
    for metric in metrics:
        print(f'test on metric : {metric}')
        data = df[df['Metric']==metric].reset_index().drop(columns=['index'])
        print(data)
        for i in range(0,3):
            datax = data.iloc[i,2:]
            for j in range(i+1,4):
                datay = data.iloc[j,2:]
                res = stats.wilcoxon(datax, datay, correction=True, alternative='greater')
                print(f'wilconx test on {models[i]} and {models[j]} : {res}')
    
    return None

if __name__ == '__main__':
    df = read_data()
    res = manntest(df)