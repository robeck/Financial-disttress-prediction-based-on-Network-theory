'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现对主要绘制程序的撰写

'''

import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay # Calibration curves 校准曲线
from matplotlib.gridspec import GridSpec

from Model.Comparasion_model import data_preprocess

def calibration_plot(clflist,traindata,preddata):
    # trainX,trainy,predX,predy = datalist
    # 参数配置
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2) # 4行2列的画布
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}

    for i, (clf, name, std_tag) in enumerate(clflist):
        trainX, trainy, predX, predy = data_preprocess(traindata,preddata,std_tag)
        clf.fit(trainX, trainy)
        display = CalibrationDisplay.from_estimator(
            clf,
            predX,
            predy,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0),(3,1)]
    for i, (_, name,std_tag) in enumerate(clflist):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass