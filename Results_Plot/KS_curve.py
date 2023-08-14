'''
Refer: https://github.com/Albertsr/Machine-Learning/blob/master/7.%20Model%20Evaluation/ks_curve.py
实现ks_curve的绘制！

'''

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plt_ks(true_y, prob_y,modelname,datetime,threshold_num=1000):

    threshold = np.linspace(np.min(prob_y),np.max(prob_y),threshold_num) # 阈值的list
    def tpr_fpr_delta(threshold):
        pred_y = np.array([int(i>threshold) for i in prob_y]) # 根据概率输出的分类结果（和阈值比较即可知道）
        tn,fp,fn,tp = confusion_matrix(true_y,pred_y).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        delta = tpr - fpr # ks的值
        return tpr,fpr,delta
    
    tprs,fprs,deltas = np.vectorize(tpr_fpr_delta)(threshold) #输出的是在不同阈值（Threshold）下的tprs，fprs，ks值
    target_tpr = tprs[np.argmax(deltas)] # KS 最大位置的索引，找到true positive rate的值
    target_fpr = fprs[np.argmax(deltas)] # KS 最大位置的索引，找到false positve rate的值
    target_threshold = threshold[np.argmax(deltas)] # 最大的那个阈值
    ks_value = np.max(deltas)
    
    "绘制部分"
    plt.figure(figsize=(8,4))
    plt.plot(threshold,tprs,label='TPR',color='r',linestyle='-',linewidth=1.5)
    plt.legend(loc='upper right')
    plt.plot(threshold,fprs,label='FPR',color='k',linestyle='-',linewidth=1.5)
    plt.legend(loc='upper right')
    plt.title(f'The KS-value curve for {modelname} on {datetime}')
    plt.xlabel('Threshold',fontsize=10)
    plt.ylabel('TPR,FPR',fontsize=10)
    plt.annotate('KS value : {:.6%}'.format(ks_value),xy=(target_threshold+0.01,0.1+0.5*ks_value))
    plt.xticks()

    "链接两个坐标点"
    x = [[target_threshold,target_threshold]]
    y = [[target_fpr,target_tpr]]

    for i in range(len(x)):
        plt.plot(x[i],y[i],'b--',lw=1.5)
        plt.scatter(x[i],y[i],c='b',s=15)
        plt.annotate('TPR : {:.6f}'.format(target_tpr), xy=([target_threshold, target_tpr]), xytext=(0.3, target_tpr),
                 arrowprops=dict(arrowstyle="<-", color='r'))
        plt.annotate('FPR : {:.6f}'.format(target_fpr), xy=([target_threshold, target_fpr]), xytext=(0.3, target_fpr),
                 arrowprops=dict(arrowstyle="<-", color='k'))
        # plt.show()

    return plt