
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV


# In[2]:


train = pd.read_csv('E:/DataMining/contest/result/train_new_feature65.csv')
test = pd.read_csv('E:/DataMining/contest/result/train_new_feature65.csv')


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
    min_data_in_leaf=40,
    num_leaves=20,
    learning_rate=0.08,
    feature_fraction=0.7,
    verbosity=-1,
    is_unbalance=True
    )


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


#测试数据与训练数据数据合并
new_feature = pd.concat([X_new_train, new_test],axis=0)


# In[16]:


# voice_opp_head_unique_count / voice_opp_num_unique_count 不同的head的对端号码占不同的对端号码的比例
new_feature['voice_unique_head_ration'] = new_feature['voice_opp_head_unique_count']/(new_feature['voice_opp_num_unique_count'])


# In[17]:


# voice_call_len1,voice_call_len2_voice_call_len3 占总通话的比例
new_feature['voice_call_len123_ratio'] = (new_feature['voice_call_len_1']+new_feature['voice_call_len_2']+new_feature['voice_call_len_3'])/new_feature['voice_opp_num_count']


# In[18]:


# 呼出/呼入
new_feature['voice_in_out_ratio'] = new_feature['voice_in_out_1']/new_feature['voice_in_out_0']


# In[19]:


# 发出短信/接收短信
new_feature['sms_in_out_ratio'] = new_feature['sms_in_out_1']/new_feature['sms_in_out_0']


# In[20]:


uid_train=pd.read_csv('E:/DataMining/contest/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))


# In[21]:


train_feature = uid_train
train_feature_new=pd.merge(train_feature,new_feature,how='left',on='uid')


# In[22]:


uid_test_new=pd.DataFrame({'uid':pd.unique(test['uid'])})


# In[23]:


test_feature = uid_test_new
test_feature_new=pd.merge(test_feature,new_feature,how='left',on='uid')


# In[24]:


train_feature_new.to_csv('E:/DataMining/contest/result/train_new_feature651.csv',index=None)
test_feature_new.to_csv('E:/DataMining/contest/result/test_new_feature651.csv',index=None)

