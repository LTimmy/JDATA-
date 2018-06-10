
# coding: utf-8

# In[1]:


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV


# In[2]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_feature651.csv')


# In[3]:


test = pd.read_csv('E:/DataMining/contest/result/test_new_feature651.csv')


# In[4]:


dtrain = xgb.DMatrix(train.drop(['uid','label'],axis=1),label=train.label)


# In[5]:


dtest = xgb.DMatrix(test.drop(['uid'],axis=1))


# In[6]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre = pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    resvalue = 0.6*auc + 0.4*f1
    return 'resvalue',resvalue


# In[7]:


xgb_params = [
    {
    'learning_rate':0.04,
    'n_estimators':1000,
    'max_depth':7,
    'min_child_weight':3,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    'nthread':4,
    'scale_pos_weight':5,
    },
]


# In[8]:


ntrain = train.shape[0] # 4999
ntest = test.shape[0] # 3000


# In[9]:


kf = KFold(n_splits=5, random_state=2018)


# In[10]:


s_train = np.zeros((ntrain,)) #1*4999
s_test = np.zeros((ntest,)) #1*3000
s_test_skf = np.empty((5,ntest)) #5*3000


# In[11]:


X_train = train.drop(['uid','label'],axis=1) # 4999*69
y_train = train.label # 4999*1
X_test = test.drop(['uid'],axis=1) # 3000*69
X_dtest = xgb.DMatrix(X_test)


# In[12]:


for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    #print(i, train_index, test_index)
    kf_X_train = X_train.iloc[train_index] # 3999 * 69
    kf_y_train = y_train.iloc[train_index] # 3999 * 1
    kf_X_test = X_train.iloc[test_index] # 1000 * 69
    kf_dtrain = xgb.DMatrix(kf_X_train,label=kf_y_train)
    kf_dtest = xgb.DMatrix(kf_X_test)
    model1_1 = xgb.train(xgb_params[0],kf_dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300)
    s_train[test_index] = model1_1.predict(kf_dtest) # 1*4999
    s_test_skf[i,:] = model1_1.predict(X_dtest)


# In[13]:


s_test[:] = s_test_skf.mean(axis=0) 


# In[14]:


uid_train=pd.read_csv('E:/DataMining/contest/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))


# In[15]:


stack_feature = pd.DataFrame()
stack_feature['stack_feature1'] = pd.Series(s_train)


# In[16]:


stack_feature['uid'] = uid_train['uid']


# In[17]:


wa_test=pd.read_csv('E:/DataMining/contest/JDATA_Test_B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name',
'visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
uid_test=pd.DataFrame({'uid':pd.unique(wa_test['uid'])})


# In[18]:


stack_test_feature = pd.DataFrame()
stack_test_feature['stack_feature1'] = pd.Series(s_test)
stack_test_feature['uid'] = uid_test['uid']


# In[19]:


stack_feature.to_csv('E:/DataMining/contest/result/xgb681_stack_train.csv',index=None)
stack_test_feature.to_csv('E:/DataMining/contest/result/xgb681_stack_test.csv',index=None)

