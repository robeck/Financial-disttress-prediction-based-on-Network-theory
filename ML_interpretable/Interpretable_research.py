'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现3个主要的解释工作

1）Permutation importance
2）PDP
3）SHAP value

'''
import eli5
from eli5.sklearn import PermutationImportance # permutation importance交换重要性

from pdpbox import pdp, get_dataset, info_plots

import shap

import matplotlib.pyplot as plt



"交换重要性绘制"
def permutation_plot(model,testX,testy,featurenames):
    perm = PermutationImportance(model,random_state=1).fit(testX,testy)
    eli5.show_weights(perm,feature_names=featurenames)
    plt.show()

    return None

"PDP: 偏依赖图"
def PDP_plot(model,testX,featurenames):
    # 逐个特征的pdp进行绘制
    for feature in featurenames:
        pdp_dist = pdp.pdp_isolate(model=model,dataset=testX,model_features=featurenames,feature=feature)
        pdp.pdp_plot(pdp_dist, feature)
        plt.show()

    return None

"SHAP value的绘制"
def SHAP_plot(model,trainX,testX):
    # 创建一个SHAP计算的对象
    explainer = shap.KernelExplainer(model.predict_proba,trainX)
    K_shape_value = explainer.shape_values(testX)
    shap.force_plot(explainer.expected_value[1],K_shape_value[1], testX)
    plt.show()

    "Advanced SHAP绘制，summary()"
    shap.summary_plot(K_shape_value[1],testX)
    plt.show()
    return None
