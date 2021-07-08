# encoding=utf-8
"""
@author: Wanziheng
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import re


#创建文件索引和标签
Data = pd.DataFrame(columns=['ID', 'SEX', 'Age'])
sex_pattern = re.compile(r'_sex  (.*?)\) \(_age')  # 编写正则 正则表达式匹配性别
age_pattern = re.compile(r'_age  (.*?)\) \(_race')# 编写正则 正则表达式匹配年龄
f1=open("face/faceDR","r")
for i in f1:#遍历每一个样本
    sex = re.findall(sex_pattern, i)#正则表达式匹配性别
    age = re.findall(age_pattern, i)#正则表达式匹配年龄
    id = i[1:5]#提取id
    #series =
    if len(sex)==0:#空值时填充Nan
        sex.append(np.nan)
    if len(age) == 0:
        age.append(np.nan)
    Data = Data.append(pd.Series({'ID':id, 'SEX':sex[0], 'Age':age[0]},),ignore_index=True)#添加记录
f2=open("face/faceDS","r")
for i in f2:
    sex = re.findall(sex_pattern, i)
    age = re.findall(age_pattern, i)
    id = i[1:5]
    #series =
    if len(sex)==0:
        sex.append(np.nan)
    if len(age) == 0:
        age.append(np.nan)
    Data = Data.append(pd.Series({'ID':id, 'SEX':sex[0], 'Age':age[0]},),ignore_index=True)
print(Data.shape)
Data=Data.dropna(how='any')
print(Data.shape)
Data.to_csv("Data.csv",index = None)
Data = pd.read_csv("Data.csv")

print(set(list(Data.SEX)))#查看性别的唯一值
print(set(list(Data.Age)))#查看年龄的唯一值
Data['SEX'] = Data['SEX'].map( {'female': 0, 'male': 1} ).astype(int)#特征编码 female用0代替
Data['Age'] = Data['Age'].map( {'child': 0, 'teen': 1,"adult":2,"senior":3} ).astype(int)

X = []
for index, row in Data.iterrows():#读取原始的人脸数据
    fid = open(f"rawdata/{row.ID}", 'rb')
    count_1 = fid.read()
    ascii = np.fromstring(count_1, dtype=np.uint8)#ascii解码到uint8
    X.append(ascii)
    fid.close()
  #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
Y_Age = list(Data['Age'])
Y_Sex = list(Data['SEX'])
#数据类型转换 list->ayyay
X_array = X[0].reshape(1,-1)
index__ = 1
for i in X[1:]:
    try:
        X_array = np.append(X_array, i.reshape(1,-1),axis=0)
    except:
        del Y_Age[index__]
        del Y_Sex[index__]
        print(i.shape)
    index__+=1
#保存与数据读取
np.save("X.npy",X_array)
np.save("Y_Age.npy",Y_Age)
np.save("Y_Sex.npy",Y_Sex)

X=np.load("X.npy",allow_pickle=True)
Y_Age=np.load("Y_Age.npy",allow_pickle=True)
Y_Sex=np.load("Y_Sex.npy",allow_pickle=True)

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()#实例化MinMax规范化
X = mm.fit_transform(X)#数据转换

from sklearn.decomposition import PCA
pca = PCA(n_components=128)#PCA降维
X = pca.fit_transform(X)

# from sklearn.model_selection import train_test_split     #导入切分训练集、测试集模块
# X_train, X_test, y_train, y_test = \
#         train_test_split(X, Y_Sex, test_size=0.2,random_state=888)#数据集划分
from sklearn.naive_bayes import GaussianNB
#网格搜索调参
from sklearn.model_selection import GridSearchCV
param_grid =  {'var_smoothing':[1e-9,1e-8,1e-10]}
nb = GaussianNB()#实例化随机森林
gsearch = GridSearchCV(estimator =nb,param_grid=param_grid,cv=5)#网格搜索调参
gsearch.fit(X,Y_Sex)
print(gsearch.best_params_)#输出最佳参数


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=7,shuffle=True)

pre = []
recall = []
auc = []
f1 = []

for i,(train_index, test_index) in enumerate(kf.split(X, Y_Sex)):
        X_train, X_test = X[train_index], X[test_index]#获取划分好的训练集和测试集
        y_train, y_test = Y_Sex[train_index], Y_Sex[test_index]
        clf1 = GaussianNB(var_smoothing=1e-9)
        clf1.fit(X_train,y_train)#模型训练
        y_pre = clf1.predict(X_test)#模型预测
        pre.append(precision_score(y_test,y_pre))#记录精度
        recall.append(recall_score(y_test, y_pre))#记录召回率
        auc.append(roc_auc_score(y_test, y_pre))#记录AUC
        f1.append(f1_score(y_test, y_pre))#记录f1值
print("sex_NB:")
print(np.array(pre).mean())
print(np.array(recall).mean())
print(np.array(auc).mean())
print(np.array(f1).mean())

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
pre = []
recall = []
auc = []
f1 = []
for i,(train_index, test_index) in enumerate(kf.split(X, Y_Sex)):
        X_train, X_test = X[train_index], X[test_index]#获取划分好的训练集和测试集
        y_train, y_test = Y_Sex[train_index], Y_Sex[test_index]
        clf2 = DecisionTreeClassifier()
        clf2.fit(X_train,y_train)
        y_pre = clf2.predict(X_test)
        pre.append(precision_score(y_test,y_pre))
        recall.append(recall_score(y_test, y_pre))
        f1.append(f1_score(y_test, y_pre))
        auc.append(roc_auc_score(y_test, y_pre))
print("sex_DT:")
print(np.array(pre).mean())
print(np.array(recall).mean())
print(np.array(auc).mean())
print(np.array(f1).mean())








pre = []
recall = []
f1 = []
for i,(train_index, test_index) in enumerate(kf.split(X, Y_Age)):
        X_train, X_test = X[train_index], X[test_index]#获取划分好的训练集和测试集
        y_train, y_test = Y_Age[train_index], Y_Age[test_index]
        clf1 = GaussianNB(var_smoothing=1e-9)
        clf1.fit(X_train,y_train)
        y_pre = clf1.predict(X_test)
        pre.append(precision_score(y_test,y_pre,average = 'macro'))
        recall.append(recall_score(y_test, y_pre,average = 'macro'))
        f1.append(f1_score(y_test, y_pre,average = 'macro'))
        #print(classification_report(y_test, y_pre,))
print("age_NB:")
print(np.array(pre).mean())
print(np.array(recall).mean())
print(np.array(f1).mean())

pre = []
recall = []
f1 = []
for i,(train_index, test_index) in enumerate(kf.split(X, Y_Age)):
        X_train, X_test = X[train_index], X[test_index]#获取划分好的训练集和测试集
        y_train, y_test = Y_Age[train_index], Y_Age[test_index]
        clf1 = DecisionTreeClassifier()
        clf1.fit(X_train,y_train)
        y_pre = clf1.predict(X_test)
        pre.append(precision_score(y_test,y_pre,average = 'macro'))
        recall.append(recall_score(y_test, y_pre,average = 'macro'))
        f1.append(f1_score(y_test, y_pre,average = 'macro'))
print("age_DT:")
print(np.array(pre).mean())
print(np.array(recall).mean())
print(np.array(f1).mean())

