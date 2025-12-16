import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader
from timefeatures import time_features
import sklearn.preprocessing
import warnings
warnings.filterwarnings('ignore')

class Dataset_Wind_Farms(Dataset):
    def __init__(self, seq_len, pred_len, flag='train', scale=False, n_cols=50):
        self.n_columns = n_cols

        self.seq_len = seq_len
        self.pred_len = pred_len
        #self.test_set_month = test_set_month

        # the last 2 months are for validation and testing respectively.
        if(flag=='train'):
            self.test_set_month = -2
        elif(flag=='test'):
            self.test_set_month = -1
        elif(flag=='validation'):
            self.test_set_month = -2

        assert flag in ['train', 'test', 'validation']
        type_map = {'train': 0, 'test': 1, 'validation': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = 0
        self.freq = 'h'
        self.data_stamp = None
        self.timeseries = None

        self.root_path = './data/AEMO'
        self.data_path = 'wind_farms.csv'
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]

        start_of_month_indexes = np.asarray(df_raw.index[df_raw['date'].str.contains('....-..-01 ..:..:..')].tolist())[::24]


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

        '''
        if self.scale:
            # Always fit on train set and transform on the other sets.
            # The 0 index means we are dealing with the test set indexes.

            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(df_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values
        '''
        data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        if (self.set_type==1):
            self.timeseries_indexes = test_set_indexes
        elif(self.set_type==0):
            # keep only the consecutive indexes (i.e. having a gap of 1).
            stepsize=1
            self.timeseries_indexes = np.split(train_set_indexes, np.where(np.diff(train_set_indexes) != stepsize)[0] + 1)[0]
        else:
            #  validation
            self.timeseries_indexes = test_set_indexes

        self.data_stamp = data_stamp
        self.timeseries = data

    def __getitem__(self, index):

        # This is called by pytorch's data loader repeatedly in order to create a random batch.
        # For example if our batch size is 32 then this function is going to be called 32 times.
        s_begin = self.timeseries_indexes[index]
        s_end = s_begin + self.seq_len + self.pred_len

        seq_x = self.timeseries[s_begin:s_end]
        seq_x = seq_x.reshape((-1, 1))

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.repeat(seq_x_mark, self.n_columns, axis=0)

        # this dataset has 50 features.
        n_feats = 50

        target_mask = np.ones_like(seq_x.reshape(self.seq_len + self.pred_len, n_feats))
        target_mask[-self.pred_len:, :] = 0

        # The future weather forecasts are all considered known values.
        # uncomment 1
        target_mask[-self.pred_len:, 10:] = 1

        s = {
            'observed_data': seq_x.reshape(self.seq_len + self.pred_len, n_feats),
            'observed_mask': np.ones_like(seq_x.reshape(self.seq_len + self.pred_len, n_feats)),
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len+self.pred_len) * 1.0,
            'timestamps': seq_x_mark[np.mod(np.arange(seq_x_mark.shape[0]),n_feats)==0],
            'feature_id': np.arange(n_feats) * 1.0,
            'farm_index_id': torch.cat((torch.arange(0, 10), torch.tensor(np.arange(10)).repeat_interleave(4)), axis=0).cpu().numpy()
        }

        return s

    def __len__(self):
        return len(self.timeseries_indexes)
