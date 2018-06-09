
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# 训练数据
uid_train=pd.read_csv('E:/DataMining/contest/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train=pd.read_csv('E:/DataMining/contest/JDATA_train/voice_train.txt',sep='\t',header=None,names=('uid',
'opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train=pd.read_csv('E:/DataMining/contest/JDATA_train/sms_train.txt',sep='\t',header=None,names=('uid',
'opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train=pd.read_csv('E:/DataMining/contest/JDATA_train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name',
'visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[3]:


# 测试数据
voice_test = pd.read_csv('E:/DataMining/contest/JDATA_Test_B/voice_test_b.txt',sep='\t',header=None,names=('uid',
'opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test=pd.read_csv('E:/DataMining/contest/JDATA_Test_B/sms_test_b.txt',sep='\t',header=None,names=('uid',
'opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test=pd.read_csv('E:/DataMining/contest/JDATA_Test_B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name',
'visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
uid_test=pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('E:/DataMining/contest/result/uid_test_b.txt',index=None)


# In[4]:


#测试数据与训练数据数据合并
voice = pd.concat([voice_train, voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)  


# In[5]:


# 通话记录
# 通话数量: voice_opp_num_count
# 不同对端号码的通话数量：voice_opp_num_unique_count
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count':lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()


# In[6]:


# 不同的对端号码前n位的数量
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count':lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index() #不同的对端号码前n位数量


# In[7]:


# 对端号码长度为3位的数量：voice_opp_len_3
# 对端号码长度为5位的数量：voice_opp_len_5
# 对端号码长度为6位的数量：voice_opp_len_6
# 对端号码长度为7位的数量：voice_opp_len_7
# 对端号码长度为8位的数量：voice_opp_len_8
# 对端号码长度为9位的数量：voice_opp_len_9
# 对端号码长度为10位的数量：voice_opp_len_10
# 对端号码长度为11位的数量：voice_opp_len_11
# .....
# # 对端号码长度为25位的数量：voice_opp_len_25
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0) #不同号码长度的用户数量


# In[8]:


# 通话类型为本地的数量：voice_call_len_1
# 通话类型为省内长途的数量：voice_call_len_2
# 通话类型为省际长途的数量：voice_call_len_3
# 通话类型为港澳台长途的数量：voice_call_len_4
# 通话类型为国际的数量：voice_call_len_5
voice_call_type=voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_len_').reset_index().fillna(0) # 不同类型的电话的数量


# In[9]:


# 呼出的电话数量 :voice_in_out_0
# 呼入的电话数量 :voice_in_out_1
voice_in_out=voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0) # 打进打出的数量


# In[10]:


def str_to_seconds(time):
    len_time = len(time)
    seconds = 0
    seconds = int(time[len_time-1])+int(time[len_time-2])*10
    minutes = 0
    minutes = int(time[len_time-3])+int(time[len_time-4])*10
    hours = 0
    hours = int(time[len_time-5])+int(time[len_time-6])*10
    seconds += (hours*60 + minutes)*60
    return seconds


# In[11]:


# 通话时长 voice_time
voice_time = pd.DataFrame()


# In[12]:


voice_time['voice_time'] = voice['end_time'].apply(str_to_seconds)-voice['start_time'].apply(str_to_seconds)


# In[13]:


voice_time['uid'] = voice['uid']


# In[14]:


# 通话时长根据时间长短分类
# 通话时长小于1分钟数量：voice_opp_num_less_than_1min
# 通话时长在1分钟到2分钟之间数量：voice_time_1min_to_2min
# 通话时长在2分钟到3分钟之间数量：voice_time_2min_to_3min
# 通话时长在3分钟到4分钟之间数量：voice_time_3min_to_4min
# 通话时长在4分钟到5分钟之间数量：voice_time_4min_to_5min
# 通话时长在5分钟到6分钟之间数量：voice_time_5min_to_6min
# 通话时长在6分钟到7分钟之间数量：voice_time_6min_to_7min
# 通话时长在7分钟到8分钟之间数量：voice_time_7min_to_8min
# 通话时长在8分钟到9分钟之间数量：voice_time_8min_to_9min
# 通话时长在9分钟到10分钟之间数量：voice_time_9min_to_10min
# 通话时长在10分到在30分钟之间数量：voice_time_10min_to_30min
# 通话时长在30分到在60分钟之间数量：voice_time_30min_to_60min
# 通话时长在60分钟以上数量：voice_time_more_than_60
voice_time_type_count = voice_time.groupby(['uid'])['voice_time'].agg({
    'less_than_1min':lambda x: len(x<60),
    '1min_to_2min':lambda x: len(x<120)-len(x<60),
    '2min_to_3min':lambda x: len(x<180)-len(x<120),
    '3min_to_4min':lambda x: len(x<240)-len(x<180),
    '4min_to_5min':lambda x: len(x<300)-len(x<240),
    '5min_to_6min':lambda x: len(x<360)-len(x<300),
    '6min_to_7min':lambda x: len(x<420)-len(x<360),
    '7min_to_8min':lambda x: len(x<480)-len(x<420),
    '8min_to_9min':lambda x: len(x<540)-len(x<480),
    '9min_to_10min':lambda x: len(x<600)-len(x<540),
    '10min_to_30min':lambda x: len(x<1800)-len(x<600),
    '30min_to_60min':lambda x: len(x<3600)-len(x<1800),
    'more_than_60min':lambda x: len(x>=3600)}).add_prefix('voice_time_').reset_index()


# In[15]:


# 短信数量：sms_opp_num_count
# 不同对端号码的短信数量：sms_opp_num_unique_count
sms_opp_num=sms.groupby(['uid'])['opp_num'].agg({'unique_count':lambda x:len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()


# In[16]:


# 不同的对端号码前n位的数量
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count':lambda x:len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()


# In[17]:


# # 对端号码长度为3位的数量：sms_opp_len_3
# 对端号码长度为5位的数量：sms_opp_len_5
# 对端号码长度为6位的数量：sms_opp_len_6
# 对端号码长度为7位的数量：sms_opp_len_7
# 对端号码长度为8位的数量：sms_opp_len_8
# 对端号码长度为9位的数量：sms_opp_len_9
# 对端号码长度为10位的数量：sms_opp_len_10
# 对端号码长度为11位的数量：sms_opp_len_11
# .....
# # 对端号码长度为25位的数量：voice_opp_len_25
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)


# In[18]:


# 发短信的数量：sms_in_out_in
# 收短信的数量：sms_in_out_out
sms_in_out=sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


# In[19]:


#sms_in_out['sms_in_out_diff'] = sms_in_out['sms_in_out_0']-sms_in_out['sms_in_out_1']


# In[20]:


# 网站/App记录


# In[21]:


# 访问网站或App总数量：wa_name_count
# 访问不同的网站或App的数量：wa_name_unique_count
wa_name=wa.groupby(['uid'])['wa_name'].agg({'unique_count':lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()


# In[22]:


# 访问网站或App的次数的一系列值
visit_cnt=wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()


# In[23]:


# 访问网站或App的总时长的一系列值
visit_dura=wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()


# In[24]:


# 访问网站或App的总上行流量的一系列值
up_flow=wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()


# In[25]:


# 访问网站或App的总下行流量的一系列值
down_flow=wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow').reset_index()


# In[26]:


feature=[voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_time_type_count,
         sms_opp_num,sms_opp_head,sms_opp_len,
        sms_in_out,wa_name,visit_cnt,visit_dura,up_flow,down_flow]


# In[27]:


train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')


# In[28]:


test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')


# In[29]:


train_feature.to_csv('E:/DataMining/contest/result/train_new_featureV1.csv',index=None)


# In[30]:


test_feature.to_csv('E:/DataMining/contest/result/test_new_featureV1.csv',index=None)

