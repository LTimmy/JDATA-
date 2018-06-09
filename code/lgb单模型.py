
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV


# In[9]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_feature651.csv')
test = pd.read_csv('E:/DataMining/contest/result/test_new_feature651.csv')


# In[10]:


dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)


# In[11]:


dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# In[12]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre = pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc + 0.4*f1
    return 'res',res,True


# In[25]:



#lg = lgb.LGBMClassifier(
    #boosting='gbdt',
    #objective='binary',
    #is_training_metric=False,
    #min_data_in_leaf=40,
    #num_leaves=20,
    #learning_rate=0.08,
    #feature_fraction=0.7,
    #verbosity=-1,
    #)
#param_test = {
    # 'num_leaves':[20, 40, 60, 80],
    # 'learning_rate':[0.02,0.04,0.06,0.08],
    # 'min_data_in_leaf':[30,35,40,45,50]
    # best chooose : 'learning_rate': 0.08, 'min_data_in_leaf': 40, 'num_leaves': 20
    #'feature_fraction':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] best:0.7
#}
#grid_search = GridSearchCV(lg, n_jobs=2, param_grid=param_test, cv=5, scoring="roc_auc", verbose=5)
#X_train = train.drop(['uid','label'],axis=1)
#y_train=train.label
#grid_search.fit( X_train, y_train)  
# last 0.90104071649673456
#grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_ 
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


# In[26]:


# 0.7894205009262958
result = lgb.cv(lgb_params, dtrain, feval=evalMetric, early_stopping_rounds=300,verbose_eval=5,
      num_boost_round=10000,nfold=3,metrics=['evalMetric'])
pd.Series(result['res-mean']).mean()


# In[27]:


model = lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300,valid_sets=[dtrain])


# In[28]:


pred=model.predict(test.drop(['uid'],axis=1))


# In[29]:


res = pd.DataFrame({'uid':test.uid,'label':pred})


# In[30]:


res=res.sort_values(by='label',ascending=False)


# In[31]:


res.label=res.label.map(lambda x:1 if x>=0.5 else 0)


# In[32]:


res.label=res.label.map(lambda x:int(x))


# In[33]:


res.to_csv('E:/DataMining/contest/result/lgb6_5_1.csv',index=False,header=False,sep=',',columns=['uid','label'])

