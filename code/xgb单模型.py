
# coding: utf-8

# In[38]:


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV


# In[39]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_feature651.csv')


# In[40]:


test = pd.read_csv('E:/DataMining/contest/result/test_new_feature651.csv')


# In[41]:


dtrain = xgb.DMatrix(train.drop(['uid','label'],axis=1),label=train.label)


# In[42]:


dtest = xgb.DMatrix(test.drop(['uid'],axis=1))


# In[43]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre = pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    resvalue = 0.6*auc + 0.4*f1
    return 'resvalue',resvalue


# In[32]:


# 确定学习速率和tree_based参数调优的估计器数目

xg1 = xgb.XGBClassifier(
    learning_rate = 0.04,
    n_estimators=1000,
    max_depth=7,
    min_child_weight=3,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=5,
    seed=27,
    reg_alpha= 0
    )
param_test1 = {
    #'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    'max_depth':[8,10,12,14,16,18]
}
grid_search = GridSearchCV(xg1, n_jobs=2, param_grid=param_test1, cv=5, scoring="roc_auc", verbose=5)
X_train = train.drop(['uid','label'],axis=1)
y_train=train.label
grid_search.fit( X_train, y_train)  
grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_ 


# In[8]:


X_train = train.drop(['uid','label'],axis=1)
y_train=train.label
xg1.fit(X_train, y_train)


# In[44]:


#0.809404
xgb_params1 = {
    'learning_rate':0.04,
    'n_estimators':1000,
    'max_depth':8,
    'min_child_weight':3,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    'nthread':4,
    'scale_pos_weight':5,
    'seed':27,
    'reg_alpha': 0
}


# In[45]:


result = xgb.cv(xgb_params1, dtrain, feval=evalMetric, early_stopping_rounds=300,verbose_eval=5,
      num_boost_round=10000,nfold=3, metrics='auc')
result


# In[46]:


model = xgb.train(xgb_params1,dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300)


# In[47]:


pred=model.predict(xgb.DMatrix(test.drop(['uid'],axis=1)))
res = pd.DataFrame({'uid':test.uid,'label':pred})
res.to_csv('E:/DataMining/contest/result/xgb_need_essembled.csv',index=None)
res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x:1 if x>=0.5 else 0)
res.label=res.label.map(lambda x:int(x))
res.to_csv('E:/DataMining/contest/result/xgb6_7_1.csv',index=False,header=False,sep=',',columns=['uid','label'])

