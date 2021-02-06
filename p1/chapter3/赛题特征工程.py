#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 数据探索
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

# 载入训练集与数据集
train_data_file = "../zhengqi_train.txt"
test_data_file = "../zhengqi_test.txt"

# 读取数据
train_data = pd.read_csv(train_data_file,sep='\t',encoding='utf-8')
test_data = pd.read_csv(test_data_file,sep='\t',encoding='utf-8')

# 查看集合的所有特征数据大撒东方大厦
train_data.head(10)


# In[3]:


# 绘制各个特征的箱线图
plt.figure(figsize=(18,10))
plt.boxplot(x=train_data.values,labels=train_data.columns) 
plt.hlines([-7.5,7.5],0,40,colors='r')
plt.show()


# In[4]:


# 绘制各个特征的箱线图
plt.figure(figsize=(18,10))
plt.boxplot(x=test_data.values,labels=test_data.columns) 
plt.hlines([-7.5,7.5],0,40,colors='r')
plt.show()


# In[6]:


# 从赛题可以分析出,v9变量存在着很大的特征异常
train_data = train_data[train_data['V9'] > -7.5]
test_data = test_data[test_data['V9'] > -7.5]
display(train_data.describe())
display(test_data.describe())


# In[7]:


# 最大最小值归一化
from sklearn import preprocessing

features_columns = [col for col in train_data.columns if col not in ['target']]
features_columns


# In[11]:


min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(train_data[features_columns])
# print(min_max_scaler)

train_data_scaler = min_max_scaler.transform(train_data[features_columns])
test_data_scaler = min_max_scaler.transform(test_data[features_columns])

train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns

test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns
train_data_scaler['target'] = train_data['target']

display(train_data_scaler.describe())
display(test_data_scaler.describe())


# In[14]:


# 根据KDE分布对比，特征变量V5,V9,V11,V17,V22,V28在训练集与测试集中差别较大，会影响模型的泛化能力，故删除这些特征
drop_col = 6
drop_row = 1

plt.figure(figsize=(5 * drop_col,5 * drop_row))

for i , col in enumerate(['V5','V9','V11','V17','V22','V28']):
    ax = plt.subplot(drop_row,drop_col,i+1)
    ax = sns.kdeplot(train_data_scaler[col],color='Red',shade=True)
    ax = sns.kdeplot(test_data_scaler[col],color='Blue',shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train","test"])
plt.show()


# In[18]:


# 计算特征相关性
plt.figure(figsize=(20,16))
column = train_data_scaler.columns.to_list()

mcorr = train_data_scaler[column].corr(method="spearman")
mask = np.zeros_like(mcorr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220,10,as_cmap=True)
g = sns.heatmap(mcorr,mask=mask,cmap=cmap,square=True,annot=True,fmt='0.2f')
plt.show()


# In[19]:


# 计算相关性系数大鱼0.1的特征变量
mcorr=mcorr.abs()
numerical_corr = mcorr[mcorr['target'] > 0.1]['target']
print(numerical_corr.sort_values(ascending=False))


# In[20]:


# 对特征变量进行排序显示
index0 = numerical_corr.sort_values(ascending=False).index
print(train_data_scaler[index0].corr('spearman'))


# In[21]:


# 多重共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 多重共线性方差膨胀因子
new_numerical = ['V0','V2','V3','V4','V5','V6','V10','V11','V13','V15','V16','V18','V19','V20','V22','V24','V30','V31','V37']

X = np.matrix(train_data_scaler[new_numerical])
VIF_list = [variance_inflation_factor(X,i) for i in range(X.shape[1])]
VIF_list


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.9)
new_train_pca_90 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_90 = pca.transform(test_data_scaler)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = train_data_scaler['target']
new_train_pca_90.describe()

