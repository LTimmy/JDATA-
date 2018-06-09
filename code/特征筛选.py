
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV


# In[2]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_featureV1.csv')
test = pd.read_csv('E:/DataMining/contest/result/test_new_featureV1.csv')


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



lg = lgb.LGBMClassifier(
    boosting='gbdt',
    objective='binary',
    is_training_metric=False,
    min_data_in_leaf=15,
    num_leaves=40,
    learning_rate=0.08,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    verbosity=-1,
    )
#grid_search = GridSearchCV(lg, n_jobs=2, param_grid=param_test, cv=5, scoring="roc_auc", verbose=5)


# In[7]:


X_train = train.drop(['uid','label'],axis=1)
y_train=train.label


# In[8]:


lg.fit(X_train, y_train)
#grid_search.fit(X_train, y_train)


# In[9]:


score = lg.feature_importances_  
print (lg.feature_importances_.shape)
#grid_search.best_estimator_


# In[10]:


score


# In[11]:


new_features = {}
i = 0
j = 0
while (i <len(score)):
    #print(score[i])
    if (score[i] > 0):
        new_features[j] = X_train.columns[i]
        j+=1
    i+=1
X_new_train = X_train[list(new_features.values())]


# In[12]:


X_new_train['uid'] = train['uid']


# In[13]:


new_test = test[list(new_features.values())]


# In[14]:


new_test['uid'] = test['uid']


# In[15]:


X_new_train.head(3)


# In[16]:


new_test.head(3)


# In[17]:


#测试数据与训练数据数据合并
new_feature = pd.concat([X_new_train, new_test],axis=0)


# In[26]:


# 短信接收与发出的差除以短信总数
new_feature['sms_in_out_diff'] = (new_feature['sms_in_out_0']-new_feature['sms_in_out_1'])/(new_feature['sms_in_out_0']+new_feature['sms_in_out_1'])


# In[27]:


# 电话呼出与呼入的差除以电话总数
new_feature['voice_in_out_diff'] = (new_feature['voice_in_out_0']-new_feature['voice_in_out_1'])/(new_feature['voice_in_out_0']+new_feature['voice_in_out_1'])


# In[28]:


# 不同的对端号码占总通话的比例
new_feature['voice_opp_num_unique_count_ratio'] = new_feature['voice_opp_num_unique_count']/new_feature['voice_opp_num_count']


# In[29]:


# 对端号码长度为12的通话占总通话的比例
new_feature['voice_opp_len12_ratio'] = new_feature['voice_opp_len_12']/new_feature['voice_opp_num_count']


# In[30]:


# 通话类型1与类型3的和占总通话数量的比例
new_feature['voice_call_len1andlen3_ratio'] = (new_feature['voice_call_len_1']+new_feature['voice_call_len_1'])/new_feature['voice_opp_num_count']


# In[31]:


# 短信对端号码长度为11与对端号码长度为13的和占短信总数量的比例
new_feature['sms_opp_len11andlen13_ratio'] = (new_feature['sms_opp_len_11']+new_feature['sms_opp_len_13'])/new_feature['sms_opp_num_count']


# In[32]:


# 上载平均值与下载平均值的差
new_feature['mean_diff_wa_up_and_wa_down'] = new_feature['wa_up_flow_mean']-new_feature['wa_down_flowmean']


# In[33]:


uid_train=pd.read_csv('E:/DataMining/contest/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))


# In[34]:


train_feature = uid_train
train_feature_new=pd.merge(train_feature,new_feature,how='left',on='uid')


# In[35]:


uid_test_new=pd.DataFrame({'uid':pd.unique(test['uid'])})


# In[36]:


test_feature = uid_test_new
test_feature_new=pd.merge(test_feature,new_feature,how='left',on='uid')


# In[37]:


train_feature_new.to_csv('E:/DataMining/contest/result/train_new_feature65.csv',index=None)
test_feature_new.to_csv('E:/DataMining/contest/result/test_new_feature65.csv',index=None)

