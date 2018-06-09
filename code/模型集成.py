
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


# In[2]:


# 读取特征
train = pd.read_csv('E:/DataMining/contest/result/train_new_feature651.csv')
test = pd.read_csv('E:/DataMining/contest/result/test_new_feature651.csv')


# In[3]:


dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# In[4]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre = pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc + 0.4*f1
    return 'res',res,True


# In[5]:


# lgb 参数设置
#0.7871405432397313
lgb_params = {
    'boosting':'gbdt',
    'objective':'binary',
    'is_training_metric':False,
    'min_data_in_leaf':40,
    'num_leaves':25,
    'learning_rate':0.05,
    'feature_fraction':0.8,
    'verbosity':-1,
    'is_unbalance':True
    #'bagging_fraction':0.7,
    #'bagging_freq':1,
}


# In[6]:


# 0.7894205009262958
result = lgb.cv(lgb_params, dtrain, feval=evalMetric, early_stopping_rounds=300,verbose_eval=5,
      num_boost_round=10000,nfold=3,metrics=['evalMetric'])
pd.Series(result['res-mean']).mean()


# In[7]:


model = lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300,valid_sets=[dtrain])


# In[8]:


pred=model.predict(test.drop(['uid'],axis=1))


# In[9]:


res = pd.DataFrame({'uid':test.uid,'label_lgb':pred})


# In[11]:


res.to_csv('E:/DataMining/contest/result/lgb_need_essembled.csv',index=None)


# In[12]:


res_xgb = pd.read_csv('E:/DataMining/contest/result/xgb_need_essembled.csv')


# In[13]:


res_lgb = pd.read_csv('E:/DataMining/contest/result/lgb_need_essembled.csv')


# In[14]:


res_essembled = pd.merge(res_lgb,res_xgb,how='left',on='uid')


# In[15]:


preds_essembled = res_essembled['label_lgb']*0.4 + res_essembled['label']*0.6


# In[16]:


last_res = pd.DataFrame({'uid':test.uid,'label':preds_essembled})


# In[17]:


last_res=last_res.sort_values(by='label',ascending=False)


# In[18]:


last_res.label=last_res.label.map(lambda x:1 if x>=0.5 else 0)


# In[19]:


last_res.label=last_res.label.map(lambda x:int(x))


# In[20]:


last_res.to_csv('E:/DataMining/contest/result/last_resultV2.csv',index=False,header=False,sep=',',columns=['uid','label'])

