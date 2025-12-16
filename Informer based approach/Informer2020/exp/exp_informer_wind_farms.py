import sys

import utils.tools
from data.data_loader import Dataset_Wind_Farms, Dataset_Wind_Farms_rand, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Wind_Farms_all_records,\
    Dataset_Wind_Farms_sliding_windows, Dataset_Wind_Farms_pick_a_month_for_test, Dataset_Wind_Farms_sliding_windows_pick_month_for_test_set, Dataset_Wind_Farms_get_test_set, Dataset_Wind_Farms_get_test_setsw
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
from utils.tools import create_plots_for_wind_farm
import glob
import random

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer_Wind_Farms(Exp_Basic):
    def __init__(self, args, settings):
        super(Exp_Informer_Wind_Farms, self).__init__(args, settings)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            #'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                # wind farms args ----------
                self.args.n_farms,
                self.args.n_cols,
                self.args.n_weather_metrics_per_farm,
                self.args.n_farms_to_predict,
                self.args.train_test_split
                #--------------
            ).float()


            # check if checkpoint exists. If so load the model end continue the training
            # else create a new informer model. The names of the checkpoint are in the format 'checkpoint_at_epoch_<i>.pth'
            # for example 'checkpoint_at_epoch_12.pth'

            path = os.path.join(self.args.checkpoints, self.settings)
            checkpoint_file_path = glob.glob(path + '/checkpoint.pth')

            if(len(checkpoint_file_path)>0):
                checkpoint_file_path = checkpoint_file_path[-1]
                print(f"Found existing checkpoint at {checkpoint_file_path}. Will continue training from where we left it ...")
                if(self.args.use_gpu == False):

                    # Uncomment this if the model was saved with DataParallel

                    # original saved file with DataParallel
                    state_dict = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)


                    #model.load_state_dict(torch.load(checkpoint_file_path, map_location=torch.device('cpu')))
                else:
                    # original saved file with DataParallel
                    state_dict = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)
                    #model.load_state_dict(torch.load(checkpoint_file_path))
            else:
                print('Did not find a checkpoint. Creating a new model ..')
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
           'WindFarms':Dataset_Wind_Farms,
           'WindFarms_rand': Dataset_Wind_Farms_rand,
           'WindFarms_sw': Dataset_Wind_Farms_sliding_windows,
           'WindFarms_pm': Dataset_Wind_Farms_pick_a_month_for_test,
           'WindFarms_pmsw': Dataset_Wind_Farms_sliding_windows_pick_month_for_test_set,
           'WindFarms_test': Dataset_Wind_Farms_get_test_set,
           'WindFarms_testsw': Dataset_Wind_Farms_get_test_setsw,


        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        elif flag == 'testtest':
            Data = Dataset_Wind_Farms_get_test_set
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'testtestsw':
            Data = Dataset_Wind_Farms_get_test_setsw
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        if(self.args.data == 'WindFarms_pm' or self.args.data == 'WindFarms_test' ):
            data_set = Data(flag=flag, scale=False, size=[args.seq_len, args.label_len, args.pred_len], n_cols=args.n_cols, test_set_month=args.test_set_month)
        elif(self.args.data == 'WindFarms_pmsw' or self.args.data == 'WindFarms_testsw'):
            data_set = Data(flag=flag, scale=False, size=[args.seq_len, args.label_len, args.pred_len], n_cols=args.n_cols, test_set_month=args.test_set_month,
                            n_preds_sliding_window=args.n_preds_sliding_window)
        else:
            data_set = Data(flag=flag, scale=False, size=[args.seq_len, args.label_len, args.pred_len], train_test_split=args.train_test_split, n_cols=args.n_cols)

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, path):
        self.model.eval()
        with torch.no_grad():
            total_loss = []
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                pred, true, enc_attn, dec_attn = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                # Create a plot for every nn_th batch
                if i % 10 == 0:
                    create_plots_for_wind_farm(path + '/plots_vali_batch_' + str(i) + '.svg',
                                               batch_x[-1].detach().cpu().numpy(),
                                               batch_y[-1].detach().cpu().numpy(),
                                               pred[-1].detach().cpu().numpy(),
                                               true[-1].detach().cpu().numpy(), self.args.n_farms_to_predict, self.args.n_cols)

                loss = criterion(pred, true)
                total_loss.append(loss.detach().cpu().numpy())
                print("\r>>> Validation: batch {}/{} loss: {}".format(i+1, vali_loader.__len__(), loss.item()), end="")
                sys.stdout.flush()

            print("\r", end="")
            sys.stdout.flush()
            val_loss = total_loss
            total_loss = np.average(total_loss)
            self.model.train()
            np.savetxt(path + '/' + 'validation_loss.csv', val_loss, delimiter=",")

            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            np.save(path + '/'+ 'val_preds.npy', preds)
            np.save(path + '/'+ 'val_trues.npy', trues)

        return total_loss

    def vali_sliding_windows(self, vali_data, vali_loader, criterion, path, window_size):
        self.model.eval()
        window_size = window_size
        with torch.no_grad():
            total_loss = []
            preds = []
            trues = []
            for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                true = batch_y[:, 0, :].reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len,
                       :self.args.n_farms].unsqueeze(0)

                # trues = torch.cat(
                #    (trues, batch_y.reshape(-1, self.args.n_cols)[self.args.label_len, :self.args.n_farms].unsqueeze(0)),
                #    0)
                enc_in = batch_x[:, 0, :]
                pred_tmp, true_tmp, _, _ = self._process_one_batch(vali_data, enc_in, batch_y[:, 0, :],
                                                                   batch_x_mark[:, 0, :], batch_y_mark[:, 0, :])

                pred = pred_tmp[:, 0, :].unsqueeze(0)

                for i in np.arange(window_size - self.args.pred_len):

                    enc_in_tmp = batch_x[:, i + 1, :]
                    # roll the tensor to shift all elements by one place to the left
                    enc_in = torch.roll(enc_in, -self.args.n_cols, 1)

                    # now copy the latest prediction to the end of enc_in
                    enc_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, -1, :self.args.n_farms] = pred[-1, :,
                                                                                                            :]

                    # fill the weather forecasts
                    enc_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, :,
                    self.args.n_farms:] = enc_in_tmp.reshape(
                        self.args.batch_size, -1, self.args.n_cols)[:, :, self.args.n_farms:]

                    dec_in = batch_y[:, i + 1, :]

                    true = torch.cat((true,
                                      dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len,
                                      :self.args.n_farms].unsqueeze(0)), 0)

                    # replace the label of the dec_in with the last self.args.label_len elements from enc_in.
                    # Note that dec_in.reshape(-1, self.args.nn_cols)[self.args.label_len:]
                    # is going to be filled with zeros by _process_one_batch()
                    dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, :self.args.label_len,
                    :] = enc_in.reshape(
                        self.args.batch_size, -1, self.args.n_cols)[:, -self.args.label_len:, :]

                    enc_mark = batch_x_mark[:, i + 1, :]
                    dec_mark = batch_y_mark[:, i + 1, :]

                    # enc_in = enc_in.reshape(-1, self.args.n_cols)[:, self.args.n_cols][-1]

                    pred_tmp, true_tmp, _, _ = self._process_one_batch(vali_data, enc_in, dec_in, enc_mark, dec_mark)

                    pred = torch.cat(
                        (pred, pred_tmp.reshape(self.args.batch_size, - 1, self.args.n_farms)[:, 0, :].unsqueeze(0)), 0)

                true = torch.cat((true, torch.permute(
                    dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len + 1:,
                    :self.args.n_farms], (1, 0, 2))), 0)
                pred = torch.cat((pred,
                                  torch.permute(pred_tmp.reshape(self.args.batch_size, -1, self.args.n_farms)[:, 1:, :],
                                                (1, 0, 2))), 0)

                true = true.type_as(pred)

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())


                loss = criterion(pred, true)
                total_loss.append(loss.detach().cpu().numpy())
                print("\r>>> Validation: batch {}/{} loss: {}".format(j+1, vali_loader.__len__(), loss.item()), end="")
                sys.stdout.flush()

            print("\r", end="")
            sys.stdout.flush()
            val_loss = total_loss
            total_loss = np.average(total_loss)
            self.model.train()
            np.savetxt(path + '/' + 'validation_loss.csv', val_loss, delimiter=",")

            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            np.save(path + '/'+ 'val_preds.npy', preds)
            np.save(path + '/'+ 'val_trues.npy', trues)

        return total_loss

    def create_plot_for_wind_farms_sliding_window(self, flag='train'):
        freq = 'h'
        preds = None
        self.args.batch_size = 1
        data_set = Dataset_Wind_Farms_all_records(
            scale=True,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            n_cols=self.args.n_cols, freq = freq)



        data_loader = Dataset_Wind_Farms_sliding_windows_pick_month_for_test_set(flag,size=[self.args.seq_len,self.args.label_len,self.args.pred_len],
                                                                                 scale=True,n_cols=self.args.n_cols,test_set_month=self.args.test_set_month,
                                                                                 n_preds_sliding_window=self.args.n_preds_sliding_window)
        index = random.randint(0, data_loader.__len__() -1)

        n_windows = self.args.n_preds_sliding_window - self.args.pred_len
        if(n_windows < 0):
            n_windows = 1

        enc_in, dec_in, enc_mark, dec_mark = data_loader.__getitem__(index)

        enc_in = torch.from_numpy(enc_in)[0].unsqueeze(0)
        dec_in = torch.from_numpy(dec_in)[0].unsqueeze(0)
        enc_mark = torch.from_numpy(enc_mark)[0].unsqueeze(0)
        dec_mark = torch.from_numpy(dec_mark)[0].unsqueeze(0)


        print(enc_in.shape)

        weather_preds = enc_in.reshape(-1, self.args.n_cols)[:, self.args.n_farms:]
        speeds_at_10 = (weather_preds.reshape(-1, 4)[:, 0] ** 2 + weather_preds.reshape(-1, 4)[:, 1] ** 2) ** 0.5
        speeds_at_10 = speeds_at_10.reshape(-1, self.args.n_farms)

        speeds_at_100 = (weather_preds.reshape(-1, 4)[:, 2] ** 2 + weather_preds.reshape(-1, 4)[:, 3] ** 2) ** 0.5
        speeds_at_100 = speeds_at_100.reshape(-1, self.args.n_farms)

        weather_preds = dec_in.reshape(-1, self.args.n_cols)[self.args.label_len, self.args.n_farms:]
        speeds_at_10 = torch.cat(
            (speeds_at_10,
             ((weather_preds.reshape(-1, 4)[:, 0] ** 2 + weather_preds.reshape(-1, 4)[:, 1] ** 2) ** 0.5).reshape(-1, self.args.n_farms)),
            0)
        speeds_at_100 = torch.cat(
            (speeds_at_100,
             ((weather_preds.reshape(-1, 4)[:, 2] ** 2 + weather_preds.reshape(-1, 4)[:, 3] ** 2) ** 0.5).reshape(-1,self.args.n_farms)),
            0)

        trues = enc_in.reshape(-1, self.args.n_cols)[:,:self.args.n_farms]

        trues = torch.cat((trues, dec_in.reshape(-1, self.args.n_cols)[self.args.label_len,:self.args.n_farms].unsqueeze(0)), 0)

        pred, true, _, _ = self._process_one_batch(None, enc_in, dec_in, enc_mark, dec_mark)

        preds = pred.reshape(-1, self.args.n_farms)[0].unsqueeze(0)

        if(n_windows == 1):

            trues = torch.cat((trues, dec_in.reshape(-1, self.args.n_cols)[self.args.label_len + 1:, :self.args.n_farms]), 0)


            preds = torch.cat((preds, pred.reshape(-1, self.args.n_farms)[1:]), 0)

            weather_preds = dec_in.reshape(-1, self.args.n_cols)[self.args.label_len + 1:, self.args.n_farms:]
            speeds_at_10 = torch.cat(
                (speeds_at_10,
                 ((weather_preds.reshape(-1, 4)[:, 0] ** 2 + weather_preds.reshape(-1, 4)[:, 1] ** 2) ** 0.5).reshape(
                     -1, self.args.n_farms)),
                0)
            speeds_at_100 = torch.cat(
                (speeds_at_100,
                 ((weather_preds.reshape(-1, 4)[:, 2] ** 2 + weather_preds.reshape(-1, 4)[:, 3] ** 2) ** 0.5).reshape(
                     -1, self.args.n_farms)),
                0)
        else:
            for i in np.arange(n_windows):
                index = index + 1

                enc_in_tmp, dec_in, enc_mark, dec_mark = data_loader.__getitem__(index)

                enc_in_tmp = torch.from_numpy(enc_in_tmp)[0].unsqueeze(0)
                dec_in = torch.from_numpy(dec_in)[0].unsqueeze(0)
                enc_mark = torch.from_numpy(enc_mark)[0].unsqueeze(0)
                dec_mark = torch.from_numpy(dec_mark)[0].unsqueeze(0)

                # roll the tensor to shift all elements by one place to the left
                enc_in = torch.roll(enc_in, -self.args.n_cols, 1)

                # now copy the latest prediction to the end of enc_in
                enc_in.reshape(-1, self.args.n_cols)[-1, :self.args.n_farms] = preds[-1, :]

                # fill the weather forecasts
                enc_in.reshape(-1, self.args.n_cols)[:, self.args.n_farms:] = enc_in_tmp.reshape(-1, self.args.n_cols)[:, self.args.n_farms:]

                enc_in.reshape(1, -1, data_set[index][0].shape[-1])


                trues = torch.cat((trues, dec_in.reshape(-1, self.args.n_cols)[self.args.label_len, :self.args.n_farms].unsqueeze(0)),0)

                # replace the label of the dec_in with the last self.args.label_len elements from enc_in.
                # Note that dec_in.reshape(-1, self.args.nn_cols)[self.args.label_len:]
                # is going to be filled with zeros by _process_one_batch()
                dec_in.reshape(-1, self.args.n_cols)[:self.args.label_len, :] = enc_in.reshape(-1, self.args.n_cols)[-self.args.label_len:, :]




                #enc_in = enc_in.reshape(-1, self.args.n_cols)[:, self.args.n_cols][-1]

                pred, true, _, _ = self._process_one_batch(None, enc_in, dec_in, enc_mark, dec_mark)

                weather_preds = dec_in.reshape(-1, self.args.n_cols)[self.args.label_len, self.args.n_farms:]
                speeds_at_10 = torch.cat(
                    (speeds_at_10,
                     ((weather_preds.reshape(-1, 4)[:, 0] ** 2 + weather_preds.reshape(-1, 4)[:,1] ** 2) ** 0.5).reshape(-1, self.args.n_farms)),
                    0)
                speeds_at_100 = torch.cat(
                    (speeds_at_100,
                     ((weather_preds.reshape(-1, 4)[:, 2] ** 2 + weather_preds.reshape(-1, 4)[:,3] ** 2) ** 0.5).reshape(-1, self.args.n_farms)),
                    0)




                preds = torch.cat((preds, pred.reshape(-1, self.args.n_farms)[0].unsqueeze(0)), 0)


            trues = torch.cat((trues, dec_in.reshape(-1, self.args.n_cols)[self.args.label_len + 1:, :self.args.n_farms]), 0)
            preds = torch.cat((preds, pred.reshape(-1, self.args.n_farms)[1:]), 0)

            weather_preds = dec_in.reshape(-1, self.args.n_cols)[self.args.label_len + 1:, self.args.n_farms:]
            speeds_at_10 = torch.cat(
                (speeds_at_10,
                 ((weather_preds.reshape(-1, 4)[:, 0] ** 2 + weather_preds.reshape(-1, 4)[:, 1] ** 2) ** 0.5).reshape(-1, self.args.n_farms)),
                0)
            speeds_at_100 = torch.cat(
                (speeds_at_100,
                 ((weather_preds.reshape(-1, 4)[:, 2] ** 2 + weather_preds.reshape(-1, 4)[:, 3] ** 2) ** 0.5).reshape(-1, self.args.n_farms)),
                0)

            print(trues.shape)
            print(preds.shape)
            print(speeds_at_10.shape)
            print(speeds_at_100.shape)

        preds_ = torch.cat((trues[self.args.seq_len - 1].unsqueeze(0) ,preds), 0)

        trues = trues.detach().numpy()
        preds = preds.detach().numpy()
        speeds_at_100 = speeds_at_100.detach().numpy()
        speeds_at_10 = speeds_at_10.detach().numpy()

        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True)

        f = 0
        labels = []
        for row in ax:
            for col in row:
                if (f < self.args.n_farms):
                    col.plot(trues[:,f], color='#7a1515')
                    col.plot(speeds_at_10[:, f], color='#7ba1c7')
                    col.plot(speeds_at_100[:, f], color='#4b7bab')
                    col.plot(np.arange(self.args.seq_len - 1, self.args.seq_len + self.args.n_preds_sliding_window), preds_.detach().numpy()[:, f], color='#0d0d8f')
                    col.set_xticks([])
                    col.set_yticks([])
                else:
                    col.set_visible(False)
                f += 1
        labels.append('power out (farms 0 - '+ str(self.args.n_farms)+')')
        labels.append('wind speed 10m')
        labels.append('wind speed 100m')
        labels.append('predictions ')
        fig.legend(labels, loc='lower right', bbox_to_anchor=(0.5,-0.01), ncol=len(labels), bbox_transform=fig.transFigure)

        fig.set_size_inches(24, 8)
        plt.savefig('./my_image.png', bbox_inches='tight')
        plt.close('all')

        return trues, preds

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        testtest_data, testtest_loader = self._get_data(flag='testtest')


        path = os.path.join(self.args.checkpoints, setting)


        if not os.path.exists(path):
            os.makedirs(path)



        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if not os.path.exists(path + '/epoch_1'):
            epoch_global_counter = 1
        else:
            epoch_global_counter = int(sorted(glob.glob(path + '/epoch_*'), key=lambda x: int(x.split('epoch_')[-1]))[-1].split('epoch_')[-1])
            epoch_global_counter = epoch_global_counter + 1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            preds = []
            trues = []

            if not os.path.exists(path + '/epoch_' + str(epoch_global_counter)):
                os.makedirs(path + '/epoch_' + str(epoch_global_counter))
            epoch_dir = path + '/epoch_' + str(epoch_global_counter)

            self.model.train()
            epoch_time = time.time()
            n_batches = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, enc_attn, dec_attn = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

                if iter_count % 100 == 0:
                    create_plots_for_wind_farm(epoch_dir + '/train_batch_' + str(iter_count) + '.svg',
                                            batch_x[-1].detach().cpu().numpy(),
                                            batch_y[-1].detach().cpu().numpy(),
                                            pred[-1].detach().cpu().numpy(),
                                            true[-1].detach().cpu().numpy(),
                                            self.args.n_farms_to_predict,
                                            self.args.n_cols)

                    # attention scores logs. Plot images with the scores. We will only plot the self attention of the decoder for the layer 0 (first).
                    utils.tools.plot_attention_scores(dec_attn[0]['self_attn'].detach().cpu().numpy(),
                                                      epoch_dir + '/train_dec_self_att_batch_' + str(
                                                        iter_count) + '.svg')

                print("\r>>> Epoch {}/{} batch {}/{} loss: {}".format(epoch+1, self.args.train_epochs, iter_count, n_batches, loss.item()), end="")
                sys.stdout.flush()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("\r", end="")
            sys.stdout.flush()
            print(">>> Epoch: {} finished after: {:.0f} seconds".format(epoch + 1, time.time() - epoch_time))
            epoch_global_counter = epoch_global_counter + 1

            # Save loss logs
            np.savetxt(epoch_dir + '/train_loss.csv', train_loss, delimiter=",")
            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, _, _ = metric(preds, trues)
            metrics = np.array([mae, mse, rmse]).reshape(1, -1)
            np.savetxt(epoch_dir + '/train_metrics.csv', metrics, delimiter=",",
                       header="mae, mse, rmse")

            np.save(epoch_dir + '/preds.npy', preds)
            np.save(epoch_dir + '/trues.npy', trues)

            train_loss = np.average(train_loss)
            test_loss = self.vali(test_data, test_loader, criterion, epoch_dir)
            if not os.path.exists(epoch_dir+ '/test_logs'):
                os.makedirs(epoch_dir+ '/test_logs')
            testtest_loss = self.vali(testtest_data, testtest_loader, criterion, epoch_dir+ '/test_logs')
            print("    train loss: {:e}".format(train_loss))
            print("    validation loss: {:e}".format(test_loss))
            print("    test loss: {:e}".format(testtest_loss))


            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # create a checkpoint at the end of each epoch
            if epoch % 1 == 0:
                print('creating a checkpoint. Saving model')
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        return self.model


    def train_sliding_windows(self, setting):
        window_size = self.args.n_preds_sliding_window
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        testtest_data, testtest_loader = self._get_data(flag='testtestsw')


        path = os.path.join(self.args.checkpoints, setting)


        if not os.path.exists(path):
            os.makedirs(path)



        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if not os.path.exists(path + '/epoch_1'):
            epoch_global_counter = 1
        else:
            epoch_global_counter = int(sorted(glob.glob(path + '/epoch_*'), key=lambda x: int(x.split('epoch_')[-1]))[-1].split('epoch_')[-1])
            epoch_global_counter = epoch_global_counter + 1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            preds = []
            trues = []

            if not os.path.exists(path + '/epoch_' + str(epoch_global_counter)):
                os.makedirs(path + '/epoch_' + str(epoch_global_counter))
            epoch_dir = path + '/epoch_' + str(epoch_global_counter)

            self.model.train()
            epoch_time = time.time()
            n_batches = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                true = batch_y[:,0,:].reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len, :self.args.n_farms].unsqueeze(0)


                #trues = torch.cat(
                #    (trues, batch_y.reshape(-1, self.args.n_cols)[self.args.label_len, :self.args.n_farms].unsqueeze(0)),
                #    0)
                enc_in = batch_x[:, 0, :]
                pred_tmp, true_tmp, _, _ = self._process_one_batch(train_data, enc_in, batch_y[:,0,:], batch_x_mark[:,0,:], batch_y_mark[:,0,:])

                pred = pred_tmp[:,0,:].unsqueeze(0)

                dec_in = None
                for i in np.arange(window_size - self.args.pred_len):

                    enc_in_tmp = batch_x[:,  i+1, :]
                    # roll the tensor to shift all elements by one place to the left
                    enc_in = torch.roll(enc_in, -self.args.n_cols, 1)

                    # now copy the latest prediction to the end of enc_in
                    enc_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, -1 , :self.args.n_farms] = pred[-1, :, :]

                    # fill the weather forecasts
                    enc_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, :, self.args.n_farms:] = enc_in_tmp.reshape(
                        self.args.batch_size,-1, self.args.n_cols)[:, :, self.args.n_farms:]

                    dec_in = batch_y[:, i+1, :]

                    true = torch.cat((true, dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len, :self.args.n_farms].unsqueeze(0)), 0)

                    # replace the label of the dec_in with the last self.args.label_len elements from enc_in.
                    # Note that dec_in.reshape(-1, self.args.nn_cols)[self.args.label_len:]
                    # is going to be filled with zeros by _process_one_batch()
                    dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, :self.args.label_len, :] = enc_in.reshape(
                                   self.args.batch_size, -1, self.args.n_cols)[:, -self.args.label_len:, :]

                    enc_mark = batch_x_mark[:,i+1,:]
                    dec_mark = batch_y_mark[:,i+1,:]

                    # enc_in = enc_in.reshape(-1, self.args.n_cols)[:, self.args.n_cols][-1]



                    pred_tmp, true_tmp, _, _ = self._process_one_batch(train_data, enc_in, dec_in, enc_mark, dec_mark)

                    pred = torch.cat((pred, pred_tmp.reshape(self.args.batch_size, - 1, self.args.n_farms)[:, 0, :].unsqueeze(0)),0)

                true = torch.cat((true, torch.permute(dec_in.reshape(self.args.batch_size, -1, self.args.n_cols)[:, self.args.label_len + 1:, :self.args.n_farms], (1, 0, 2))), 0)
                pred = torch.cat((pred, torch.permute(pred_tmp.reshape(self.args.batch_size, -1, self.args.n_farms)[:, 1:, :], (1, 0, 2))), 0)


                true = true.type_as(pred)

                loss = criterion(pred, true)
                train_loss.append(loss.item())
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())


                print("\r>>> Epoch {}/{} batch {}/{} loss: {}".format(epoch+1, self.args.train_epochs, iter_count, n_batches, loss.item()), end="")
                sys.stdout.flush()
                break
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("\r", end="")
            sys.stdout.flush()
            print(">>> Epoch: {} finished after: {:.0f} seconds".format(epoch + 1, time.time() - epoch_time))
            epoch_global_counter = epoch_global_counter + 1

            # Save loss logs
            np.savetxt(epoch_dir + '/train_loss.csv', train_loss, delimiter=",")
            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, _, _ = metric(preds, trues)
            metrics = np.array([mae, mse, rmse]).reshape(1, -1)
            np.savetxt(epoch_dir + '/train_metrics.csv', metrics, delimiter=",",
                       header="mae, mse, rmse")

            np.save(epoch_dir + '/preds.npy', preds)
            np.save(epoch_dir + '/trues.npy', trues)

            train_loss = np.average(train_loss)
            test_loss = self.vali_sliding_windows(test_data, test_loader, criterion, epoch_dir, window_size)
            if not os.path.exists(epoch_dir+ '/test_logs'):
                os.makedirs(epoch_dir+ '/test_logs')
            testtest_loss = self.vali_sliding_windows(testtest_data, testtest_loader, criterion, epoch_dir+ '/test_logs', window_size)

            print("    train loss: {:e}".format(train_loss))
            print("    validation loss: {:e}".format(test_loss))
            print("    test loss: {:e}".format(testtest_loss))


            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # create a checkpoint at the end of each epoch
            if epoch % 1 == 0:
                print('creating a checkpoint. Saving model')
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        return self.model


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true, enc_attn, dec_attn = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'preds.npy', preds)
        np.save(folder_path + 'trues.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        #print("--------------------------")
        #print(torch.cuda.device_count())
        #print(torch.cuda.current_device())
        #print("--------------------------")

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        n_cols = self.args.n_cols
        n_farms = self.args.n_farms
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len * n_cols, batch_y.shape[-1]]).float()
            dec_inp.reshape((-1, n_cols))[:, n_farms:n_cols] = batch_y[:, self.args.label_len * n_cols:, :].reshape((-1, n_cols))[:, n_farms:n_cols]
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len * n_cols, batch_y.shape[-1]]).float()
            dec_inp.reshape((-1, n_cols))[:, n_farms:n_cols] = batch_y[:, self.args.label_len * n_cols:, :].reshape((-1, n_cols))[:, n_farms:n_cols]
        dec_inp = torch.cat([batch_y[:,:self.args.label_len*n_cols,:], dec_inp], dim=1).float().to(self.device)


        enc_attn = None
        dec_attn = None

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs, enc_attn, dec_attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs, enc_attn, dec_attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


        batch_y = batch_y.reshape(
            (self.args.batch_size, self.args.pred_len + self.args.label_len, n_cols)
        )[:, self.args.label_len:, :self.args.n_farms_to_predict].to(self.device)

        return outputs, batch_y, enc_attn, dec_attn

    def get_model(self):
        return self.model
