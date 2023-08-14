'''
@Author: haozhi chen
@Date: 2022-08
@Target: 实现以下基于粒子群优化的SVM模型

Utile: 一般是进行模型数据的预处理，进行绘制工作的
'''
from sklearn.metrics import ConfusionMatrixDisplay #进行绘制的包
import matplotlib.pyplot as plt


'''对confusion矩阵的绘制
'''
def confusion_martrix_disp(classifer,x_test,y_test,dislabel,title_options):
    '''
    :param classifer:
    :param x_test:
    :param y_test:
    :param dislabel:
    :param title_options:
    :return:
    '''

    for title,normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifer,
            x_test,
            y_test,
            display_labels=dislabel,
            cmap = plt.cm.Blues,
            normalize=normalize
        )

        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()
    return None


def plot(position):
    x,y = [],[]
    for i in range(len(position)):
        x.append(position[i][0])
        y.append(position[i][1])
    colors = (0,0,0)
    plt.scatter(x,y,c=colors,alpha=0.1)
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.axis([0,10,0,10])
    plt.gca().set_aspect('equal',adjustable='box')
    return plt.show()


if __name__ == '__main__':
    pass
