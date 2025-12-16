import os
import numpy as np
import pandas as pd

import math

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
import sklearn.preprocessing
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        scale = False
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Wind_Farms_pick_a_month_for_test(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, n_cols=50, test_set_month=-1):
        self.n_columns = n_cols

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.test_set_month = test_set_month

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        start_of_month_indexes = np.asarray(df_raw.index[df_raw['date'].str.contains('....-..-01 ..:..:..')].tolist())[::24]
        print(start_of_month_indexes.shape)

        # if you want the last month of the dataset you may use -1
        month_pick_for_test_set = self.test_set_month
        test_set_start = start_of_month_indexes[month_pick_for_test_set]
        test_set_end = None
        if(test_set_start == start_of_month_indexes[-1]):
            test_set_end = df_raw.shape[0] -self.seq_len -self.pred_len
        else:
            test_set_end = start_of_month_indexes[month_pick_for_test_set + 1] -self.seq_len -self.pred_len + 1


        all_indexes_size = df_raw.shape[0]
        all_timestamps_indexes = np.arange(all_indexes_size)
        all_timestamps_indexes = all_timestamps_indexes[:-self.seq_len - self.pred_len + 1]
        if(month_pick_for_test_set == -1):
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[:(test_set_indexes[0] - self.seq_len + 1)]
        elif(month_pick_for_test_set == 0):
            test_set_indexes = all_timestamps_indexes[:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[(test_set_indexes[-1] +self.seq_len):-self.seq_len -self.pred_len + 1]
        else:
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = np.delete(all_timestamps_indexes, np.s_[(test_set_indexes[0] - self.seq_len + 1):(test_set_indexes[-1] +self.seq_len)], 0)

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        if (self.set_type):
            self.timeseries_indexes = test_set_indexes
        else:
            self.timeseries_indexes = train_set_indexes

        self.data_stamp = data_stamp
        self.timeseries = data

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = self.timeseries_indexes[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_y = self.timeseries[r_begin:r_end]
        seq_y = seq_y.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
        seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.timeseries_indexes)

class Dataset_Wind_Farms_get_test_set(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, n_cols=50, test_set_month=-1):
        self.n_columns = n_cols

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.test_set_month = -1
        flag = 'test'
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/complete_dataset'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        start_of_month_indexes = np.asarray(df_raw.index[df_raw['date'].str.contains('....-..-01 ..:..:..')].tolist())[::24]
        print(start_of_month_indexes.shape)

        # if you want the last month of the dataset you may use -1
        month_pick_for_test_set = self.test_set_month
        test_set_start = start_of_month_indexes[month_pick_for_test_set]
        test_set_end = None
        if(test_set_start == start_of_month_indexes[-1]):
            test_set_end = df_raw.shape[0] -self.seq_len -self.pred_len
        else:
            test_set_end = start_of_month_indexes[month_pick_for_test_set + 1] -self.seq_len -self.pred_len + 1


        all_indexes_size = df_raw.shape[0]
        all_timestamps_indexes = np.arange(all_indexes_size)
        all_timestamps_indexes = all_timestamps_indexes[:-self.seq_len - self.pred_len + 1]
        if(month_pick_for_test_set == -1):
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[:(test_set_indexes[0] - self.seq_len + 1)]
        elif(month_pick_for_test_set == 0):
            test_set_indexes = all_timestamps_indexes[:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[(test_set_indexes[-1] +self.seq_len):-self.seq_len -self.pred_len + 1]
        else:
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = np.delete(all_timestamps_indexes, np.s_[(test_set_indexes[0] - self.seq_len + 1):(test_set_indexes[-1] +self.seq_len)], 0)

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        if (self.set_type):
            self.timeseries_indexes = test_set_indexes
        else:
            self.timeseries_indexes = train_set_indexes

        self.data_stamp = data_stamp
        self.timeseries = data

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = self.timeseries_indexes[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_y = self.timeseries[r_begin:r_end]
        seq_y = seq_y.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
        seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.timeseries_indexes)






class Dataset_Wind_Farms(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, train_test_split=0.7, n_cols=50):
        self.n_columns = n_cols

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_test_split = train_test_split

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Split the dataset in train test. The first math.floor(len(df_raw) * self.train_test_split) elements will be the train set
        # and the rest will be the test set.

        border1s = [
            0,
            math.floor(len(df_raw) * self.train_test_split) - self.seq_len
        ]

        border2s = [
            math.floor(len(df_raw) * self.train_test_split),
            len(df_raw)
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the standard scaler")
            train_data = df_data[border1s[0]:border2s[0]]
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.timeseries = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_y = self.timeseries[r_begin:r_end]
        seq_y = seq_y.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
        seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.timeseries) - self.seq_len- self.pred_len + 1


class Dataset_Wind_Farms_sliding_windows(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, train_test_split=0.7, n_cols=50):
        self.n_columns = n_cols
        # This is the total number of predictions using sliding windows. Note that this is not the number of predictions
        # of the decoder's output. Normally, this will be larger than self.pred_len (decoder's output).
        self.n_preds = 12
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_test_split = train_test_split

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Split the dataset in train test. The first math.floor(len(df_raw) * self.train_test_split) elements will be the train set
        # and the rest will be the test set.

        border1s = [
            0,
            math.floor(len(df_raw) * self.train_test_split) - self.seq_len
        ]

        border2s = [
            math.floor(len(df_raw) * self.train_test_split),
            len(df_raw)
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the standard scaler")
            train_data = df_data[border1s[0]:border2s[0]]
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.timeseries = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # the decoder produces self.args.pred_len predictions
        # with the sliding window we want to predict a total of self.n_preds predictions
        #
        seq_x_windows = []
        seq_y_windows = []
        seq_x_mark_windows = []
        seq_y_mark_windows = []
        for i in np.arange(self.n_preds - self.pred_len + 1):
            # This is called by pytorch's data loader repeatedly in order to create a random batch.
            # For example if our batch size is 32 then this function is going to be called 32 times.
            s_begin = index + i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.timeseries[s_begin:s_end]
            seq_x = seq_x.reshape((-1, 1))

            seq_y = self.timeseries[r_begin:r_end]
            seq_y = seq_y.reshape((-1, 1))

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

            seq_y_mark = self.data_stamp[r_begin:r_end]
            # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
            seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


            seq_x_windows.append(seq_x)
            seq_y_windows.append(seq_y)
            seq_x_mark_windows.append(seq_x_mark)
            seq_y_mark_windows.append(seq_y_mark)

        seq_x_windows = np.array(seq_x_windows)
        seq_y_windows = np.array(seq_y_windows)
        seq_x_mark_windows = np.array(seq_x_mark_windows)
        seq_y_mark_windows = np.array(seq_y_mark_windows)


        return seq_x_windows, seq_y_windows, seq_x_mark_windows, seq_y_mark_windows

    def __len__(self):
        return len(self.timeseries) - self.seq_len -self.pred_len -(self.n_preds - self.pred_len) + 1

class Dataset_Wind_Farms_sliding_windows_pick_month_for_test_set(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, n_cols=50, test_set_month=-1, n_preds_sliding_window=24):
        self.n_columns = n_cols
        # This is the total number of predictions using sliding windows. Note that this is not the number of predictions
        # of the decoder's output. Normally, this will be larger than self.pred_len (decoder's output).
        self.n_preds = n_preds_sliding_window
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.test_set_month = test_set_month

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]


        start_of_month_indexes = np.asarray(df_raw.index[df_raw['date'].str.contains('....-..-01 ..:..:..')].tolist())[
                                 ::24]
        print(start_of_month_indexes.shape)

        # if you want the last month of the dataset you may use -1
        month_pick_for_test_set = self.test_set_month
        test_set_start = start_of_month_indexes[month_pick_for_test_set]
        test_set_end = None
        if (test_set_start == start_of_month_indexes[-1]):
            test_set_end = df_raw.shape[0] - self.seq_len - self.pred_len
        else:
            test_set_end = start_of_month_indexes[month_pick_for_test_set + 1] - self.seq_len - self.pred_len + 1

        all_indexes_size = df_raw.shape[0]
        all_timestamps_indexes = np.arange(all_indexes_size)
        all_timestamps_indexes = all_timestamps_indexes[:-self.seq_len - self.n_preds + 1]
        if (month_pick_for_test_set == -1):
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[:(test_set_indexes[0] - self.seq_len -(self.n_preds - self.pred_len) + 1)]
        elif (month_pick_for_test_set == 0):
            test_set_indexes = all_timestamps_indexes[:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[
                                (test_set_indexes[-1] + self.seq_len):-self.seq_len - self.n_preds + 1]
        else:
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = np.delete(all_timestamps_indexes, np.s_[
                                                                  (test_set_indexes[0] - self.seq_len -(self.n_preds - self.pred_len)+ 1):
                                                                  (test_set_indexes[-1] + self.seq_len)], 0)
        test_set_indexes = test_set_indexes[:-(self.n_preds - self.pred_len)]

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        if self.scale:
            # Normally we fit on test set and transform on other sets.
            # Because we are dealing with weather predictions and power outputs of a system whose
            # technical specifications are supposedely known we assume that we can just fit_transform
            # on the entire dataset.

            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        if (self.set_type):
            self.timeseries_indexes = test_set_indexes
        else:
            self.timeseries_indexes = train_set_indexes

        print(train_set_indexes.shape)
        print(test_set_indexes.shape)

        self.data_stamp = data_stamp
        self.timeseries = data



    def __getitem__(self, index):
        # the decoder produces self.args.pred_len predictions
        # with the sliding window we want to predict a total of self.n_preds predictions
        #
        index = self.timeseries_indexes[index]
        seq_x_windows = []
        seq_y_windows = []
        seq_x_mark_windows = []
        seq_y_mark_windows = []
        for i in np.arange(self.n_preds - self.pred_len + 1):
            # This is called by pytorch's data loader repeatedly in order to create a random batch.
            # For example if our batch size is 32 then this function is going to be called 32 times.
            s_begin = index + i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.timeseries[s_begin:s_end]
            seq_x = seq_x.reshape((-1, 1))

            seq_y = self.timeseries[r_begin:r_end]
            seq_y = seq_y.reshape((-1, 1))

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

            seq_y_mark = self.data_stamp[r_begin:r_end]
            # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
            seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


            seq_x_windows.append(seq_x)
            seq_y_windows.append(seq_y)
            seq_x_mark_windows.append(seq_x_mark)
            seq_y_mark_windows.append(seq_y_mark)

        seq_x_windows = np.array(seq_x_windows)
        seq_y_windows = np.array(seq_y_windows)
        seq_x_mark_windows = np.array(seq_x_mark_windows)
        seq_y_mark_windows = np.array(seq_y_mark_windows)


        return seq_x_windows, seq_y_windows, seq_x_mark_windows, seq_y_mark_windows

    def __len__(self):
        return len(self.timeseries_indexes)


class Dataset_Wind_Farms_get_test_setsw(Dataset):
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, n_cols=50, test_set_month=-1, n_preds_sliding_window=24):
        self.n_columns = n_cols
        # This is the total number of predictions using sliding windows. Note that this is not the number of predictions
        # of the decoder's output. Normally, this will be larger than self.pred_len (decoder's output).
        self.n_preds = n_preds_sliding_window
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.test_set_month = -1
        flag='test'
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/complete_dataset'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]


        start_of_month_indexes = np.asarray(df_raw.index[df_raw['date'].str.contains('....-..-01 ..:..:..')].tolist())[
                                 ::24]
        print(start_of_month_indexes.shape)

        # if you want the last month of the dataset you may use -1
        month_pick_for_test_set = self.test_set_month
        test_set_start = start_of_month_indexes[month_pick_for_test_set]
        test_set_end = None
        if (test_set_start == start_of_month_indexes[-1]):
            test_set_end = df_raw.shape[0] - self.seq_len - self.pred_len
        else:
            test_set_end = start_of_month_indexes[month_pick_for_test_set + 1] - self.seq_len - self.pred_len + 1

        all_indexes_size = df_raw.shape[0]
        all_timestamps_indexes = np.arange(all_indexes_size)
        all_timestamps_indexes = all_timestamps_indexes[:-self.seq_len - self.n_preds + 1]
        if (month_pick_for_test_set == -1):
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[:(test_set_indexes[0] - self.seq_len -(self.n_preds - self.pred_len) + 1)]
        elif (month_pick_for_test_set == 0):
            test_set_indexes = all_timestamps_indexes[:test_set_end + 1]
            train_set_indexes = all_timestamps_indexes[
                                (test_set_indexes[-1] + self.seq_len):-self.seq_len - self.n_preds + 1]
        else:
            test_set_indexes = all_timestamps_indexes[test_set_start:test_set_end + 1]
            train_set_indexes = np.delete(all_timestamps_indexes, np.s_[
                                                                  (test_set_indexes[0] - self.seq_len -(self.n_preds - self.pred_len)+ 1):
                                                                  (test_set_indexes[-1] + self.seq_len)], 0)
        test_set_indexes = test_set_indexes[:-(self.n_preds - self.pred_len)]

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        if self.scale:
            # Normally we fit on test set and transform on other sets.
            # Because we are dealing with weather predictions and power outputs of a system whose
            # technical specifications are supposedely known we assume that we can just fit_transform
            # on the entire dataset.

            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        if (self.set_type):
            self.timeseries_indexes = test_set_indexes
        else:
            self.timeseries_indexes = train_set_indexes

        print(train_set_indexes.shape)
        print(test_set_indexes.shape)

        self.data_stamp = data_stamp
        self.timeseries = data



    def __getitem__(self, index):
        # the decoder produces self.args.pred_len predictions
        # with the sliding window we want to predict a total of self.n_preds predictions
        #
        index = self.timeseries_indexes[index]
        seq_x_windows = []
        seq_y_windows = []
        seq_x_mark_windows = []
        seq_y_mark_windows = []
        for i in np.arange(self.n_preds - self.pred_len + 1):
            # This is called by pytorch's data loader repeatedly in order to create a random batch.
            # For example if our batch size is 32 then this function is going to be called 32 times.
            s_begin = index + i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.timeseries[s_begin:s_end]
            seq_x = seq_x.reshape((-1, 1))

            seq_y = self.timeseries[r_begin:r_end]
            seq_y = seq_y.reshape((-1, 1))

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

            seq_y_mark = self.data_stamp[r_begin:r_end]
            # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
            seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


            seq_x_windows.append(seq_x)
            seq_y_windows.append(seq_y)
            seq_x_mark_windows.append(seq_x_mark)
            seq_y_mark_windows.append(seq_y_mark)

        seq_x_windows = np.array(seq_x_windows)
        seq_y_windows = np.array(seq_y_windows)
        seq_x_mark_windows = np.array(seq_x_mark_windows)
        seq_y_mark_windows = np.array(seq_y_mark_windows)


        return seq_x_windows, seq_y_windows, seq_x_mark_windows, seq_y_mark_windows

    def __len__(self):
        return len(self.timeseries_indexes)






class Dataset_Wind_Farms_rand(Dataset):
    '''
     Because the dataset is relatively limited we opted first for the approach used by the class Dataset_Wind_Farms.
     In Dataset_Wind_Farms the sampling is done sequentially for the time steps of the time series. This way we can have
     the maximum datasets for true and pred when taking into account the size. This is because the overlap of the timestamps
     is the minimum.

     The problem with this approach is that the test set contains the last n timestamps. This is inconvenient. We would like
     to have a random sampling, that makes more sense. But the cost of this is that the train dataset is going to be smaller
     because of some eventual overlapping between the train and test timestamps. All such timestamps must be disqualified from
     the train dataset.
    '''
    def __init__(self, flag='train',
                 size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, train_test_split=0.7, n_cols=50):
        self.n_columns = n_cols

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_test_split = train_test_split

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]


        # create an array with all the indexes of the dataset
        all_indexes_size = df_raw.shape[0]
        all_timestamps_indexes = np.arange(all_indexes_size)

        # remove all indexes that lead out of the bounds of the dataset
        for idx, b in enumerate(all_timestamps_indexes):
            if(b > (all_indexes_size - self.seq_len - self.pred_len)):
                all_timestamps_indexes[idx] = -1

        all_timestamps_indexes = all_timestamps_indexes[all_timestamps_indexes != -1]


        # in order to not get a significantly reduced or possibly empty train set, we must use a small number here. About 100 - 200 is recommended.
        test_set_size = 800

        # set the seed in order to get same results every time
        # do not change the seed nor the test_set_size if you want to resume training of the model.
        seed = 32

        np.random.seed(seed)

        # these are the indexes for the test set
        test_set_indexes = np.random.choice(all_timestamps_indexes, size=test_set_size, replace=False)


        # remove all indices of the train set that have been disqualified for having an overlap with the test set
        for b in test_set_indexes:
            for idx, a in enumerate(all_timestamps_indexes):
                if a <=  (b + self.pred_len + self.seq_len - 1) and a > (b - self.seq_len):
                    # mark index as disqualified if the index is in the test set and it lies inside the context of a train set sequences.
                    all_timestamps_indexes[idx] = -1

        # Delete the disqualified indexes
        train_set_indexes = all_timestamps_indexes[all_timestamps_indexes != -1]

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_stamp = data_stamp
        self.timeseries = data

        if (self.set_type):
            self.timeseries_indexes = test_set_indexes
        else:
            self.timeseries_indexes = train_set_indexes

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = self.timeseries_indexes[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_y = self.timeseries[r_begin:r_end]
        seq_y = seq_y.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
        seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.timeseries_indexes)


class Dataset_Wind_Farms_all_records(Dataset):
    '''
     Because the dataset is relatively limited we opted first for the approach used by the class Dataset_Wind_Farms.
     In Dataset_Wind_Farms the sampling is done sequentially for the time steps of the time series. This way we can have
     the maximum datasets for true and pred when taking into account the size. This is because the overlap of the timestamps
     is the minimum.

     The problem with this approach is that the test set contains the last n timestamps. This is inconvenient. We would like
     to have a random sampling, that makes more sense. But the cost of this is that the train dataset is going to be smaller
     because of some eventual overlapping between the train and test timestamps. All such timestamps must be disqualified from
     the train dataset.
    '''
    def __init__(self, size=[24*5, # seq_len: Number of timesteps in the future to use as a context for our predictions.
                       # seq_len is related to the encoder's input
                       24*2, # label_len: This is the number of past timesteps to be appended to the begining of the decoder's input.
                       # Instead of starting the decoder's input sequence with a special 'start' token we use a fraction of
                       # the actual context sequence. The reason for apending this sequence is because of the nature of the
                       # Informer's decoder. The vanilla Transformer predicts only one future timestep prediction for each forward pass
                       # The Informer's decoder produces all the predictions for all future timesteps at once.
                       24*1 # pred_len: this is the number of future predictions we want.
                       # Note that the decoder's inpout is going to be of size label_len + pred_len in terms of timesteps.
                       ],
                 scale=True, n_cols=50, freq='h'):
        self.n_columns = n_cols

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/WIND_FARMS_DATASET/'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.
            print("Scaling dataset using the MinMaxScaler scaler")
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_stamp = data_stamp
        self.timeseries = data

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_y = self.timeseries[r_begin:r_end]
        seq_y = seq_y.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 50: 10 power outputs for the 10 farms plus 4 weather forecasts * 10 farms
        seq_y_mark = np.repeat(seq_y_mark, self.n_columns, axis=0)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.timeseries) - self.seq_len- self.pred_len + 1
