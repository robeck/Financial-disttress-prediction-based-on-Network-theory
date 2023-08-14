'''
@Author: haozhic
@Date: 2022-07
@Target: 实现对DEA分析的简单工作


'''

import gurobipy # 这个gurobipy有很多自带的函数，需要阅读说明文档
import pandas as pd
import numpy as np
import openpyxl

import scipy.optimize as op # scipy下的优化求解器

# 分页显示数据, 设置为 False 不允许分页
pd.set_option('display.expand_frame_repr', False)

# 最多显示的列数, 设置为 None 显示全部列
pd.set_option('display.max_columns', None)

# 最多显示的行数, 设置为 None 显示全部行
pd.set_option('display.max_rows', None)


class DEA(object):
    def __init__(self, data,DMUs_Name, X, Y, AP=False):
        '''
        :param data: 数据集，便于SBM运算的
        :param DMUs_Name:
        :param X:
        :param Y:
        :param AP:
        '''
        self.data = data
        self.m1, self.m1_name, self.m2, self.m2_name, self.AP = \
            X.shape[1], X.columns.tolist(), Y.shape[1], Y.columns.tolist(), AP
        ''':arg
        m1: 投入项的个数
        m1_name: 投入项的名称
        m2：产出项的个数
        m2_name：产出项名称
        AP：
        '''
        self.DMUs, self.X, self.Y = gurobipy.multidict(
            {DMU: [X.loc[DMU].tolist(), Y.loc[DMU].tolist()] for DMU in DMUs_Name}) #一键多值字典。其实其结构类似于一个复合多类型数据的列表
        '''
        DMUs - [0] : index，决策单元，每一行就是一个决策单元！
        X - [1] : X 投入项的字典
        Y - [2] : Y 产出项的字典
        '''
        print(f'DEA(AP={AP}) MODEL RUNING...')

    def __CCR(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            OE, lambdas, s_negitive, s_positive = MODEL.addVar(), MODEL.addVars(self.DMUs), MODEL.addVars(
                self.m1), MODEL.addVars(self.m2)
            MODEL.update()
            MODEL.setObjectiveN(OE, index=0, priority=1)
            MODEL.setObjectiveN(-(sum(s_negitive) + sum(s_positive)), index=1, priority=0)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) + s_negitive[
                    j] == OE * self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) - s_positive[
                    j] == self.Y[k][j] for j in range(self.m2))
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()
            self.Result.at[k, ('效益分析', '综合技术效益(CCR)')] = MODEL.objVal
            self.Result.at[k, ('规模报酬分析',
                               '有效性')] = '非 DEA 有效' if MODEL.objVal < 1 else 'DEA 弱有效' if s_negitive.sum().getValue() + s_positive.sum().getValue() else 'DEA 强有效'
            self.Result.at[k, ('规模报酬分析',
                               '类型')] = '规模报酬固定' if lambdas.sum().getValue() == 1 else '规模报酬递增' if lambdas.sum().getValue() < 1 else '规模报酬递减'
            for m in range(self.m1):
                self.Result.at[k, ('差额变数分析', f'{self.m1_name[m]}')] = s_negitive[m].X
                self.Result.at[k, ('投入冗余率', f'{self.m1_name[m]}')] = 'N/A' if self.X[k][m] == 0 else s_negitive[m].X / \
                                                                                                     self.X[k][m]
            for m in range(self.m2):
                self.Result.at[k, ('差额变数分析', f'{self.m2_name[m]}')] = s_positive[m].X
                self.Result.at[k, ('产出不足率', f'{self.m2_name[m]}')] = 'N/A' if self.Y[k][m] == 0 else s_positive[m].X / \
                                                                                                     self.Y[k][m]
        return self.Result

    def __BCC(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            TE, lambdas = MODEL.addVar(), MODEL.addVars(self.DMUs)
            MODEL.update()
            MODEL.setObjective(TE, sense=gurobipy.GRB.MINIMIZE)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) <= TE *
                self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) >= self.Y[k][j]
                for j in range(self.m2))
            MODEL.addConstr(gurobipy.quicksum(lambdas[i] for i in self.DMUs if i != k or not self.AP) == 1)
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()

            self.Result.at[
                k, ('效益分析', '技术效益(BCC)')] = MODEL.objVal if MODEL.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
        return self.Result

    def __CRS(self):
        # 存储数据的字典
        E = {}
        stkcdlist = []
        valuelist = []
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            v,u = {},{}
            for i in range(self.m1):
                v[k,i] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name="v_%s%d"%(k,i),lb=0.0001)
            for j in range(self.m2):
                u[k,j] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name="u_%s%d"%(k,j),lb=0.0001)
            MODEL.update()
            MODEL.setObjective(gurobipy.quicksum(u[k,j]*self.Y[k][j] for j in range(self.m2)),gurobipy.GRB.MAXIMIZE)
            MODEL.addConstr(gurobipy.quicksum(v[k,i]*self.X[k][i] for i in range(self.m1))==1)
            for r in self.DMUs:
                MODEL.addConstr(gurobipy.quicksum(u[k,j]*self.Y[r][j] for j in range(self.m2))-gurobipy.quicksum(v[k,i]*self.X[r][i] for i in range(self.m1))<=0)
            MODEL.setParam('OutputFlag',0) # 不让求解过程进行输出！
            MODEL.optimize()
            "try-excepy部分是为了让程序持续的运行下去，后续会让这些报错的值=0即可"
            try:
                error_raise_value = MODEL.objVal # 用于唤起错误，避免后面出现数据长度不匹配的问题
                # print(f"The efficiency of DMU {k} is {MODEL.objVal}") # 验证在另一个数据集下面是有好的结果
                stkcdlist.append(k)
                valuelist.append(MODEL.objVal)
            except (AttributeError):
                print(f"Errors on the efficiency of DMU {k} is 0")
                stkcdlist.append(k)
                valuelist.append(0)


        E['Stkcd'] = stkcdlist
        E['effeciency'] = valuelist
        print(len(E.get('Stkcd')))
        print(len(E.get('effeciency')))
        return E

    def __VRS(self):
        # 初始化存储的数据
        E = {}
        stkcdlist = []
        valuelist = []
        for k in self.DMUs:
            MODEL = gurobipy.Model('VRS')
            v,u,u0 = {},{},{}
            for i in range(self.m1):
                v[k,i] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"v_{k}{i}",lb=0.0001)
            for j in range(self.m2):
                u[k,j] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"u_{k}{j}",lb=0.0001)
            u0[k] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"u_0{k}",lb=-1000)
            MODEL.update()
            MODEL.setObjective(gurobipy.quicksum(u[k,j]*self.Y[k][j] for j in range(self.m2))-u0[k],gurobipy.GRB.MAXIMIZE)
            MODEL.addConstr(gurobipy.quicksum(v[k,i] * self.X[k][i] for i in range(self.m1))==1)
            for r in self.DMUs:
                MODEL.addConstr(gurobipy.quicksum(u[k,j]*self.Y[r][j] for j in range(self.m2))-gurobipy.quicksum(v[k,i]*self.X[r][i] for i in range(self.m1))-u0[k] <= 0)
            MODEL.setParam('OutputFlag',0)
            MODEL.optimize()
            "try-except 部分是为了让程序运行下去，报错的结果会显示为0，后期让输出的值=0"
            try:
                # print(f"The efficiency of DMU {k} is {MODEL.objVal}")
                # print(f"The u0{u0[k].varName} : {u0[k].X}")
                error_raise_value = MODEL.objVal
                stkcdlist.append(k)
                valuelist.append(MODEL.objVal)
            except (AttributeError):
                print(f"Errors on the efficiency of DMU {k} is 0")
                print(f"The u0{u0[k].varName} : {u0[k].X}")
                stkcdlist.append(k)
                valuelist.append(0)
                continue
        E['Stkcd'] = stkcdlist
        E['effeciency'] = valuelist
        return E


    ''':Slack-based measures
    考虑了松弛变量的方法！
    增加features：
    （1）非期望产出：undesirable_output
    （2）方法：revised_samplex interior-point
    '''
    def __SBM_undesirable(self,desirable_varibale, undesirable_variable,method):
        '''
        :param undesirable_output: 期望的输出变量
        :param method:  方法'revised simplex' or 'interior-point'
        :param desirbale_flag: 设置为期望，非期望的标签
        :return: 我们返回的都是dictionary
        '''
        # 初始化一个存户的
        E = {}
        dmu = self.DMUs # DMU 就是每一行的索引标签
        input_variable = self.m1_name # 投入项目的名称
        output_variable = self.m2_name # 包含了全部的expected variable和unexpected variable
        E['Stkcd'] = dmu
        # dmu单元个数
        dum_num = len(dmu)
        # 投入的变量个数
        input_num = len(input_variable)
        # 产出（期望产出）的个数
        expoutput_num = len(desirable_varibale)
        # 产出（非期望）个数
        unexpoutput_num  = len(undesirable_variable)

        total = dum_num + input_num + expoutput_num + unexpoutput_num + 1 # 表示，行数+列数+1。具体含义需要考虑
        columns = input_variable + output_variable # 全部的列

        newcol = [] # 存储新的columns，即增加的slack变量
        total_efficient = [] # 统计效率指标
        stocklist = [] #统计index（这里是stock）
        # for j in columns:
        #     newcol.append(j+'_slcak')
        #     E[j+'_slcak'] = np.nan

        print('############################ SBM model 正在运行 ###################################')
        "SBM整个优化过程"
        for i in self.DMUs: # 循环索引比较好
            "优化目标"
            c = [0] * dum_num + [1] + list(-1/(input_num + self.data.loc[i,input_variable])) + [0] * (expoutput_num+unexpoutput_num)

            "约束条件：约束方程的系数矩阵"
            A_eq = [[0] * dum_num + [1] + [0] * input_num + list(1/((expoutput_num + unexpoutput_num)*self.data.loc[i,desirable_varibale]))+
                                                            list(1/((expoutput_num + unexpoutput_num)*self.data.loc[i,undesirable_variable]))]

            "约束条件（1）：投入松弛变量为正"
            for j1 in range(input_num):
                list1 = [0] * input_num
                list1[j1] = 1
                eq1 = list(self.data[input_variable[j1]]) + [-self.data.loc[i,input_variable[j1]]] + list1 + [0] * (expoutput_num + unexpoutput_num)
                A_eq.append(eq1)

            "约束条件（2）：期望产出松弛变量为正"
            for j2 in range(expoutput_num):
                list2 = [0] * expoutput_num
                list2[j2] = -1
                eq2 = list(self.data[desirable_varibale[j2]]) + [-self.data.loc[i,desirable_varibale[j2]]] + [0]*input_num + list2 + [0]*unexpoutput_num
                A_eq.append(eq2)

            "约束条件（3）：非期望产出松弛变量为正"
            for j3 in range(unexpoutput_num):
                list3 = [0] * unexpoutput_num
                list3[j3] = 1
                eq3 = list(self.data[undesirable_variable[j3]]) + [-self.data.loc[i,undesirable_variable[j3]]] + [0] * (input_num + expoutput_num) + list3
                A_eq.append(eq3)

            "约束条件：常量"
            b_eq = [1] + [0] * (input_num + expoutput_num + unexpoutput_num)
            bounds = [(0,None)] * total # 约束边界为0

            print(A_eq)
            "求解过程"
            op1 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method=method)
            total_efficient.append(op1.fun)
            stocklist.append(i)

        E['Stkcd'] = stocklist
        E['total_efficient'] = total_efficient
        return E

    ''':针对desirable variable
    不考虑非期望产出的情况
    
    '''
    def __SBM_desirable(self,desirable_varibale,method):
        '''
        :param undesirable_output: 期望的输出变量
        :param method:  方法'revised simplex' or 'interior-point'
        :param desirbale_flag: 设置为期望，非期望的标签
        :return: 我们返回的都是dictionary
        '''
        # 初始化一个存户的
        E = {}
        dmu = self.DMUs # DMU 就是每一行的索引标签
        input_variable = self.m1_name # 投入项目的名称
        output_variable = self.m2_name # 包含了全部的expected variable和unexpected variable
        E['Stkcd'] = dmu
        # dmu单元个数
        dum_num = len(dmu)
        # 投入的变量个数
        input_num = len(input_variable)
        # 产出（期望产出）的个数
        expoutput_num = len(desirable_varibale)


        total = dum_num + input_num + expoutput_num  + 1 # 表示，行数+列数+1。具体含义需要考虑
        columns = input_variable + output_variable # 全部的列

        newcol = [] # 存储新的columns，即增加的slack变量
        total_efficient = [] # 统计效率指标
        stocklist = [] #统计index（这里是stock）
        # for j in columns:
        #     newcol.append(j+'_slcak')
        #     E[j+'_slcak'] = np.nan

        print('############################ SBM model 正在运行 ###################################')
        "SBM整个优化过程"
        for i in self.DMUs: # 循环索引比较好
            "优化目标"
            c = [0] * dum_num + [1] + list(-1/(input_num + self.data.loc[i,input_variable])) + [0] * (expoutput_num)

            "约束条件：约束方程的系数矩阵"
            A_eq = [[0] * dum_num + [1] + [0] * input_num + list(1/((expoutput_num)*self.data.loc[i,desirable_varibale]))]

            "约束条件（1）：投入松弛变量为正"
            for j1 in range(input_num):
                list1 = [0] * input_num
                list1[j1] = 1
                eq1 = list(self.data[input_variable[j1]]) + [-self.data.loc[i,input_variable[j1]]] + list1 + [0] * (expoutput_num)
                A_eq.append(eq1)

            "约束条件（2）：期望产出松弛变量为正"
            for j2 in range(expoutput_num):
                list2 = [0] * expoutput_num
                list2[j2] = 1
                eq2 = list(self.data[desirable_varibale[j2]]) + [-self.data.loc[i,desirable_varibale[j2]]] + [0]*input_num + list2
                A_eq.append(eq2)


            "约束条件：常量"
            b_eq = [1] + [0] * (input_num + expoutput_num)
            bounds = [(0,None)] * total # 约束边界为0

            "求解过程"
            op1 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method=method)
            total_efficient.append(op1.fun)
            stocklist.append(i)

        E['Stkcd'] = stocklist
        E['total_efficient'] = total_efficient
        return E


    def dea(self,method,desirable_variable,undesirable_variable):
        res = []
        # columns_Page = ['效益分析'] * 4 + ['规模报酬分析'] * 2 + ['差额变数分析'] * (self.m1 + self.m2) + ['投入冗余率'] * self.m1 + [
        #     '产出不足率'] * self.m2
        # columns_Group = ['技术效益(BCC)', '规模效益(CCR/BCC)', '综合技术效益(CCR)','总体效率值(OE)', '有效性', '类型'] + (self.m1_name + self.m2_name) * 2
        # '''
        # columns_page : 代表的是第一层的columns
        # columns_group ： 代表的是第二层的columns
        # '''
        #
        # self.Result = pd.DataFrame(index=self.DMUs, columns=[columns_Page, columns_Group])

        # self.__CCR()
        # self.__BCC()
        if method == 'CRS':
            res = self.__CRS() # 新增加的模型
        elif method == 'VRS':
            res = self.__VRS() # 新增加的模型
        elif method == 'undes_SBM':
            opmethod = 'revised simplex'
            res = self.__SBM_undesirable(desirable_variable,undesirable_variable,opmethod)
        elif method == 'des_SBM':
            opmethod = 'revised simplex'
            res = self.__SBM_desirable(desirable_variable,opmethod)

        # self.Result.loc[:, ('效益分析', '规模效益(CCR/BCC)')] = self.Result.loc[:, ('效益分析', '综合技术效益(CCR)')] / self.Result.loc[:,
        #                                                                                               ('效益分析',
        #                                                                                     '技术效益(BCC)')]
        return res

    def analysis(self,method,desirable_variable,undesirable_variable,file_name=None):
        Result = self.dea(method,desirable_variable,undesirable_variable)
        # file_name = 'DEA 数据包络分析报告.xlsx' if file_name is None else f'\\{file_name}.xlsx'
        # Result.to_excel(file_name, 'DEA 数据包络分析报告')
        return Result


'''
测试主函数
'''
def main():
    # 初始一个可调整的method
    method = []
    
    "测试数据集1"
    method = 'VRS'
    data = pd.DataFrame({1990: [14.40, 0.65, 31.30, 3621.00, 0.05], 1991: [16.90, 0.72, 32.20, 3943.00, 0.09],
                         1992: [15.53, 0.72, 31.87, 4086.67, 0.07], 1993: [15.40, 0.76, 32.23, 4904.67, 0.13],
                         1994: [14.17, 0.76, 32.40, 6311.67, 0.37], 1995: [13.33, 0.69, 30.77, 8173.33, 0.59],
                         1996: [12.83, 0.61, 29.23, 10236.00, 0.51], 1997: [13.00, 0.63, 28.20, 12094.33, 0.44],
                         1998: [13.40, 0.75, 28.80, 13603.33, 0.58], 1999: [14.00, 0.84, 29.10, 14841.00, 1.00]},
                        index=['政府财政收入占 GDP 的比例/%', '环保投资占 GDP 的比例/%', '每千人科技人员数/人', '人均 GDP/元', '城市环境质量指数']).T

    X = data[['政府财政收入占 GDP 的比例/%', '环保投资占 GDP 的比例/%', '每千人科技人员数/人']]  #截取的dataframe
    Y = data[['人均 GDP/元', '城市环境质量指数']]
    desirable_variable = ['人均 GDP/元']
    undesirable_variable = ['城市环境质量指数']
    print(X)
    print(Y)
    
    dea = DEA(data=data,DMUs_Name=data.index, X=X, Y=Y)
    res = dea.analysis(method,None,None)  # dea 分析并输出表格
    print(res)
    "测试SBM方法"
    method = 'undes_SBM'
    res = dea.analysis(method,desirable_variable,undesirable_variable)
    print(res)
    method = 'des_SBM'
    desirable_variable = ['人均 GDP/元', '城市环境质量指数']
    res = dea.analysis(method,desirable_variable,None)
    print(res)
    # print(dea.dea())  # dea 分析，不输出结果
    
    "测试数据集2"
    method = 'VRS'
    # "用于测试CRS模型的数据"
    data = pd.DataFrame({'A':[11,14,2,2,1],'B':[7,7,1,1,1],'C':[11,14,1,1,2],'D':[14,14,2,3,1],'E':[14,15,3,2,3]},
                        index=['x1','x2','y1','y2','y3']).T
    X = data[['x1','x2']]
    Y = data[['y1','y2','y3']]

    dea = DEA(data=data,DMUs_Name=data.index,X=X,Y=Y)
    res = dea.analysis(method,None,None)
    print(res)




if __name__ == '__main__':
    main()
