
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold


# In[2]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_feature651.csv')
test = pd.read_csv('E:/DataMining/contest/result/test_new_feature651.csv')


# In[3]:


dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)


# In[4]:


dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# In[5]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre = pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc + 0.4*f1
    return 'res',res,True


# In[6]:



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
lgb_params = [
    {
    'boosting':'gbdt',
    'objective':'binary',
    'is_training_metric':False,
    'min_data_in_leaf':40,
    'num_leaves':20,
    'learning_rate':0.08,
    'feature_fraction':0.8,
    'verbosity':-1,
    'is_unbalance':True,
    'seed':27
    },
    {
    'boosting':'gbdt',
    'objective':'binary',
    'is_training_metric':False,
    'min_data_in_leaf':40,
    'num_leaves':20,
    'learning_rate':0.08,
    'feature_fraction':0.8,
    'verbosity':-1,
    'is_unbalance':True,
    'seed':9999
    },
    {
    'boosting':'gbdt',
    'objective':'binary',
    'is_training_metric':False,
    'min_data_in_leaf':40,
    'num_leaves':20,
    'learning_rate':0.08,
    'feature_fraction':0.8,
    'verbosity':-1,
    'is_unbalance':True
    },
    
]


# In[7]:


ntrain = train.shape[0] # 4999
ntest = test.shape[0] # 3000
kf = KFold(n_splits=5, random_state=2018)
s_train = np.zeros((ntrain,)) #1*4999
s_test = np.zeros((ntest,)) #1*3000
s_test_skf = np.empty((5,ntest)) #5*3000


# In[8]:


X_train = train.drop(['uid','label'],axis=1) # 4999*69
y_train = train.label # 4999*1
X_test = test.drop(['uid'],axis=1) # 3000*69
X_dtest = lgb.Dataset(X_test)


# In[9]:


for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    #print(i, train_index, test_index)
    kf_X_train = X_train.iloc[train_index] # 3999 * 69
    kf_y_train = y_train.iloc[train_index] # 3999 * 1
    kf_X_test = X_train.iloc[test_index] # 1000 * 69
    kf_dtrain = lgb.Dataset(kf_X_train,label=kf_y_train)
    #kf_dtest = lgb.Dataset(kf_X_test)
    model1_1 = lgb.train(lgb_params[0],kf_dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300)
    model1_2 = lgb.train(lgb_params[1],kf_dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300)
    model1_3 = lgb.train(lgb_params[2],kf_dtrain,feval=evalMetric,verbose_eval=5,
                 num_boost_round=300)
    pred1 = model1_1.predict(kf_X_test)
    pred2 = model1_2.predict(kf_X_test)
    pred3 = model1_3.predict(kf_X_test)
    pred = (pred1+pred2+pred3)/3
    s_train[test_index] = pred # 1*4999
    pred1_test = model1_1.predict(X_test)
    pred2_test = model1_2.predict(X_test)
    pred3_test = model1_3.predict(X_test)
    pred_test = (pred1_test+pred2_test+pred3_test)/3
    s_test_skf[i,:] = pred_test


# In[10]:


s_test[:] = s_test_skf.mean(axis=0) 


# In[11]:


from sklearn.linear_model import LogisticRegression as LR


# In[12]:


bclf = LR()


# In[13]:


s_train = s_train.reshape(-1, 1)


# In[14]:


bclf.fit(s_train, y_train)


# In[15]:


Y_test_predict = bclf.predict()


# In[16]:


uid_train=pd.read_csv('E:/DataMining/contest/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))


# In[17]:


stack_feature = pd.DataFrame()
stack_feature['stack_feature2'] = pd.Series(s_train)
stack_feature['uid'] = uid_train['uid']


# In[ ]:


stack_feature.info()


# In[ ]:


wa_test=pd.read_csv('E:/DataMining/contest/JDATA_Test_B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name',
'visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
uid_test=pd.DataFrame({'uid':pd.unique(wa_test['uid'])})


# In[ ]:


stack_test_feature = pd.DataFrame()
stack_test_feature['stack_feature2'] = pd.Series(s_test)
stack_test_feature['uid'] = uid_test['uid']


# In[ ]:


stack_feature.to_csv('E:/DataMining/contest/result/lgb_stack_train.csv',index=None)
stack_test_feature.to_csv('E:/DataMining/contest/result/lgb_stack_test.csv',index=None)

