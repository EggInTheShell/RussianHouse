import numpy as np
import pandas as pd
import pickle

#ここにかいとく
train_data1 = pd.read_csv('rawdata/train_new_non_categorical.csv')
train1 = np.array(train_data1.as_matrix())
start_date = '2000-01-01'
traindata = np.zeros_like(train1, dtype=np.float32)
traindata[:, 2] = (pd.to_datetime(train1[:,2])-pd.to_datetime(start_date)).days
traindata[:, 0:2] = train1[:, 0:2]
traindata[:, 3:] = train1[:, 3:]

train_data2 = pd.read_csv('rawdata/train_new_categorical_dummy.csv')
train2 = np.array(train_data2.as_matrix()).astype(np.float32)
train_all = np.concatenate([traindata, train2],axis=1)

#データの標準化
train_copy = np.copy(train_all)
train_std = (train_copy - train_copy.mean()) / train_copy.std()


#save
savepath = 'train_std.pkl'
with open(savepath, mode='wb') as f:
   pickle.dump(train_std, f)

#testも同様
test_data1 = pd.read_csv('rawdata/test_new_non_categorical.csv')
test1 = np.array(test_data1.as_matrix())
start_date = '2000-01-01'
testdata = np.zeros_like(test1, dtype=np.float32)
testdata[:, 2] = (pd.to_datetime(test1[:,2])-pd.to_datetime(start_date)).days
testdata[:, 0:2] = test1[:, 0:2]
testdata[:, 3:] = test1[:, 3:]

test_data2 = pd.read_csv('rawdata/test_new_categorical_dummy.csv')
test2 = np.array(test_data2.as_matrix()).astype(np.float32)
test_all = np.concatenate([testdata, test2],axis=1)


#データの標準化
test_copy = np.copy(test_all)
test_std = (test_copy - train_copy.mean()) / train_copy.std()

savepath = 'test_std.pkl'
with open(savepath, mode='wb') as f:
   pickle.dump(test_std, f)