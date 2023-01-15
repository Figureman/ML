
import numpy as np
#建立Ann模型

class Ann:
    def __init__(self,inputLayer,hiddenLayer,outputLayer,eta):
        '''
        :param inputLayer:  输入层 假设初始为4
        :param hiddenLayer: 隐藏层 假设初始为6
        :param outputLayer: 输出层 假设初始为3
        :param eta:         学习率
        '''
        #参数初始化
        self.inputLayer=inputLayer
        self.hiddenLayer=hiddenLayer
        self.outputLayer=outputLayer
        self.eta=eta

        #变量初始化
        # 输入层 -> 隐藏层
        self.w1=np.random.normal(0.0, pow(hiddenLayer, -0.5), (hiddenLayer, inputLayer))   #权重 w1初始化为 【6,4】矩阵 正态分布
        self.b1=np.zeros((hiddenLayer,1))                                                  #权重 b1初始化为 【6,1】矩阵，全为0

        # 隐藏层  -> 输出层
        self.w2=np.random.normal(0.0, pow(outputLayer, -0.5), (outputLayer, hiddenLayer))   #权重 w2初始化为 【3,6】矩阵 正态分布
        self.b2=np.zeros((outputLayer,1))                                                  #权重 b2初始化为 【3,1】矩阵，全为0


    #定义激活函数
    def Activatefun(self,x):
        return 1.0/(1.0+np.exp(-x))

    #前向传播
    def forward(self):
        #隐藏层输出值【6,1】
        self.output1=self.Activatefun(np.dot(self.w1,self.data)+self.b1)
        #输出层输出值【3,1】
        self.output2=self.Activatefun(np.dot(self.w2,self.output1)+self.b2)

    #反向传播
    def backward(self):
        #g更新  【3,1】
        self.g=self.output2*(1-self.output2)*(self.label-self.output2)
        #e更新 【6,1】
        self.e=self.output1*(1-self.output1)*np.dot(self.w2.T,self.g)

    #梯度计算
    def gred(self):
        self.w2_gred=np.dot(self.g,self.output1.T)
        self.b2_gred=-self.g
        self.w1_gred=np.dot(self.e,self.data.T)
        self.b1_gred=-self.e

    #梯度更新
    def update(self):
        self.w1+=self.w1_gred
        self.b1+=self.b1_gred
        self.w2+=self.w2_gred
        self.b2+=self.b2_gred

    #Ann模型训练
    def train(self,data,label):
        self.data=np.array(data,ndmin=2).T      #【4,1】
        self.label=np.array(label,ndmin=2).T    #【3,1】
        self.forward()
        self.backward()
        self.gred()
        self.update()

    #求模型的精确度以及收集未分类正确的数据
    def accuracy(self,data,labels):
        self.data = np.array(data, ndmin=2).T
        self.forward()
        labels_true=np.argmax(labels,axis=1)
        labels_pred=np.argmax(self.output2.T,axis=1)
        #求准确率
        rate=sum(labels_pred==labels_true)/labels.shape[0]
        #收集分类错误的样本
        error_index = np.arange(0, len(labels_true))
        error_index = error_index[labels_pred != labels_true]       # 得到未正确分类数据的下标
        return rate,labels_pred,labels_true,error_index

