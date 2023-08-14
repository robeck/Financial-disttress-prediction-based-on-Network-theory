'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现对GBDT 梯度下降决策树生成新特征 加入 就特征并使用LR进行预测的展示

'''

import numpy as np
from scipy.sparse import hstack
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier # GBDT 算法
from sklearn.linear_model import LogisticRegression # logistic regression 算法
from sklearn.svm import SVC # SVC
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm


'''
GBDT + LR
1) GBDT 生成新的特征
2）LR 进行预测

此外，我们在 4 中测试了GBDT+SVC
结果显示：其效果非常的好
'''
def GBDT_LR_test():
    # 生成实验数据
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=18, n_redundant=2,
                               n_classes=2, n_clusters_per_class=3, random_state=2017)
    trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.3,random_state=108)
    print(trainX)
    print(trainX.shape)
    
    "1. 直接训练，测试GBDT"
    clf = GradientBoostingClassifier(n_estimators=50) # 参数没有优化
    clf.fit(trainX,trainy)
    y_pred = clf.predict(testX)
    y_prob = clf.predict_proba(testX)[:,1] # predict_proba()返回的是一个多列的数据。如果是二分类问题，则 0 列表示的是预测为（0）的概率。1 列表示的是预测为（1）的概率
    acc = accuracy_score(testy,y_pred)
    auc = roc_auc_score(testy,y_prob)
    print('Original features')
    print(f'ACC得分为：{acc}')
    print(f'AUC得分为：{auc}')
    print(f'原始训练数据的特征维度为：{trainX.shape}，原始测试数据的特征维度为：{testX.shape}')

    "2. 测试生成新特征"
    '''
    核心思想：将Tree的节点，通过OneHot编译成0-1特征，作为输入参数加入到数据集的特征中来
    
    使用训练好的树模型构造特征
    (1)使用RandomForestClassifier自带的apply接口可以很轻松的拿到叶子节点的ID
    (2)然后通过OneHotEncoder对叶子节点的ID进行onehot编码就可以构造出叶子节点特征了。
    '''
    "生成每个样本在每棵树叶节点的索引军阵"
    X_train_leaves = clf.apply(trainX)[:,:,0]
    X_test_leaves = clf.apply(testX)[:,:,0]
    "OneHotEncoder操作"
    enc = OneHotEncoder(categories='auto')
    "fit-transform出新的特征出来"
    X_train_new = enc.fit_transform(X_train_leaves) # 这里拟合，转换
    X_test_new =enc.transform(X_test_leaves) # 这里只需要转换即可，因为已经使用训练数据拟合了
    print(f'训练数据新特征的维度：{X_train_new.shape},测试数据特征的维度：{X_test_new.shape}')
    "新旧特征的结合"
    X_train_merge = hstack([trainX,X_train_new])
    X_test_merge = hstack([testX,X_test_new])

    "3. 将新特征使用到LR预测中"
    lr = LogisticRegression(solver='lbfgs',max_iter=1000)
    lr.fit(X_train_merge,trainy)
    y_pred = lr.predict(X_test_merge)
    y_prob = lr.predict_proba(X_test_merge)[:,1]
    "计算各项得分"
    acc = accuracy_score(testy,y_pred)
    auc = roc_auc_score(testy,y_prob)
    print("加入New features之后，使用LR预测")
    print(f'ACC得分：{acc}')
    print(f'AUC得分：{auc}')

    "4 额外测试，SVC预测"
    svc = SVC(gamma='auto',probability=True) # probability=True才能输出概率，计算auc得分
    svc.fit(X_train_merge,trainy)
    y_pred = svc.predict(X_test_merge)
    y_prob = svc.predict_proba(X_test_merge)[:,1]
    "计算各项得分"
    acc = accuracy_score(testy,y_pred)
    auc = roc_auc_score(testy,y_prob)
    print("加入New features之后，使用SVC进行预测")
    print(f'ACC得分：{acc}')
    print(f'AUC得分：{auc}')


    return None

"我们只做特征提取是否可以呢，让其生成新的特征出来"
def GBDT(train_testdata,preddata):
    trainX, trainy,testX, testy = train_testdata
    predX, predy = preddata
    print(f'训练数据的维度：{trainX.shape},{trainy.shape},测试数据集的维度：{testX.shape},{testy.shape},预测数据集的维度：{predX.shape}，{predy.shape}')

    # 训练，测试数据的划分
    # trainX,testX,trainy,testy = train_test_split(dataX,datay,test_size=0.3,random_state=90)

    "为了更好的应用模型，我们需要对参数进行优化"
    "1.使用学习曲线来选择合适的n_estimators参数"
    scores = []
    for i in tqdm(range(0,200,10)): #循环0-200，间隔10。这些都是estimators的设置
        model = GradientBoostingClassifier(n_estimators=i+1,
                                           random_state=0)
        score = cross_val_score(model,trainX,trainy,cv=10).mean() # 交叉验证均分
        scores.append(score)
    best_scores_1 = max(scores)
    best_n_estimators = (scores.index(max(scores))*10)+1
    print(f'训练数据最优得分：{best_scores_1}, 其参数为{best_n_estimators}') #输出最优的得分，并且给出这个对应的n_estimators
    "2.一次性gridsearchcv的方法，寻找最优的参数"
    parameters = {'criterion':['friedman_mse', 'squared_error'],'max_depth':np.arange(1,30,1),'min_samples_leaf':np.arange(1,20,1)}
    model = GradientBoostingClassifier(n_estimators=best_n_estimators,
                                       random_state=0)
    GS = GridSearchCV(model,parameters,cv=10)
    GS.fit(trainX,trainy)
    best_scores_2 = GS.best_score_
    best_parameters = GS.best_params_
    print(f'一次性gridsearch搜索criterion，max depth，min samples leaf的最佳得分为：{best_scores_2},最佳参数为：{best_parameters}')

    "选择合适的模型"
    # if best_scores_2 > best_scores_1:
    #     best_criterion = parameters.get('criterion')
    #     best_max_depth = parameters.get('max_depth')
    #     best_min_sample_leaf = parameters.get("min_samples_leaf")
    #     best_clf = GradientBoostingClassifier(n_estimators=best_n_estimators,
    #                                           criterion=best_criterion,
    #                                           max_depth=best_max_depth,
    #                                           min_samples_leaf=best_min_sample_leaf,
    #                                           random_state=0)
    # else:
    best_clf = GradientBoostingClassifier(n_estimators=best_n_estimators,random_state=0)

    "最优GDBT模型的拟合，结果输出"
    best_clf.fit(trainX,trainy)
    predy = best_clf.predict(testX)
    proby = best_clf.predict_proba(testX)[:,1]
    acc = accuracy_score(testy,predy)
    auc = roc_auc_score(testy,proby)
    print(f'最佳GBDT模型在original features下的得分')
    print(f'ACC得分：{acc}')
    print(f'AUC得分：{auc}')

    "2. 输出GBDT给到的new features"
    trainX_leaf = best_clf.apply(trainX)[:,:,0]
    testX_leaf = best_clf.apply(testX)[:,:,0]
    predX_leaf = best_clf.apply(predX)[:,:,0]
    "数据的Onehotencoder"
    '''
    有相关工作提到：要现在axis=0方向上合并数据集的叶节点索引，在进行oneHotencoder操作
    + 避免多彩OneHotEncoder操作生成的系数矩阵列数不等
    '''
    All_leaves = np.r_[trainX_leaf,testX_leaf,predX_leaf]
    "OneHotEncoder：生成新特征"
    enc = OneHotEncoder(categories='auto')
    new_features = enc.fit_transform(All_leaves)

    "拆分train，test，pred"
    train_samples = trainX.shape[0]
    test_samples = train_samples + testX.shape[0] # test数据的位置是（train列 - （train+test）列）
    pred_samples = test_samples + predX.shape[0] # pred数据的位置是（（train+test）列 - （最后））
    trainX_new = new_features[:train_samples,:]
    testX_new = new_features[train_samples:test_samples,:]
    predX_new = new_features[test_samples:,:]

    "合并新旧特征"
    trainX_merge = hstack([trainX,trainX_new])
    testX_merge = hstack([testX,testX_new])
    predX_merge = hstack([predX,predX_new])
    print(f'新和成的训练数据维度：{trainX_merge.shape}，测试数据维度：{testX_merge.shape}，预测数据维度:{predX_merge.shape}')

    ''':return
    合并了新features的
    TrainX
    TestX
    PredictX
    '''
    return trainX_merge,trainy,testX_merge,testy,predX_merge,predy


if __name__ == '__main__':
    GBDT_LR_test()