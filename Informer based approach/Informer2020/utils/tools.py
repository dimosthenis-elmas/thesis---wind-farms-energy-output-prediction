import numpy as np
import torch
import math
import os
import pandas as pd
from pandas import DataFrame
import glob
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        #lr = lr_adjust[epoch]
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            if os.path.exists(path + '/best_validation_score.npy'):
                self.best_score = max(np.load(path + '/best_validation_score.npy'), score)
            else:
                self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            np.save(path + '/best_validation_score.npy', np.array([self.best_score]))
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            np.save(path + '/best_validation_score.npy', np.array([self.best_score]))
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            print('Creating checkpoint for best model in validation.')
            if not os.path.exists(path + '/best_in_val'):
                os.makedirs(path + '/best_in_val')
            torch.save(model.state_dict(), path + '/best_in_val/' + 'checkpoint_val_'+ str(-self.best_score).replace('.','_')[:-4] + '.pth')
            #torch.save(model.state_dict(), path + '/checkpoint_best_in_validation/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def create_plots_for_wind_farm(path, enc_in ,dec_in ,pred, true, n_farms, n_cols):
    t = []
    p = []
    s_10 = []
    s_100 = []
    n_preds = pred.shape[-2]
    enc_in = enc_in.reshape(-1, n_cols)
    n_context = enc_in.shape[-2]
    for farm in np.arange(n_farms):
        dec_in_ = dec_in.reshape(-1, n_cols)[:pred.shape[-2]]
        weather_preds = np.concatenate((enc_in, dec_in_), axis=0)
        weather_preds = weather_preds[:, n_farms:n_cols][:, farm + 0: farm + 4]
        pred_ = pred[:, farm]
        true_ = true[:, farm]
        true_ = np.concatenate((enc_in[:, 0:n_farms][:, farm], true_), axis=0)

        speeds_at_10 = (weather_preds[:, 0]**2 + weather_preds[:, 1]**2)** 0.5
        speeds_at_100 = (weather_preds[:, 2] ** 2 + weather_preds[:, 3] ** 2) ** 0.5

        #scaler = sklearn.preprocessing.StandardScaler()
        #true_ = scaler.fit_transform(true_.reshape(-1, 1))
        #scaler = sklearn.preprocessing.StandardScaler()
        #pred_ = scaler.fit_transform(pred_.reshape(-1, 1))
        scaler = sklearn.preprocessing.MinMaxScaler()
        speeds_at_10 = scaler.fit_transform(speeds_at_10.reshape(-1, 1))
        scaler = sklearn.preprocessing.MinMaxScaler()
        speeds_at_100 = scaler.fit_transform(speeds_at_100.reshape(-1, 1))

        t.append(true_)
        p.append(pred_)
        s_10.append(speeds_at_10)
        s_100.append(speeds_at_100)


    span1 = np.arange(n_context, n_context + n_preds)
    span2 = np.arange(0, n_context + n_preds)
    span3 = np.arange(0, n_preds)

    fig, ax = plt.subplots(nrows=2, ncols=5)


    f = 0
    for row in ax:
        for col in row:
            if (f < n_farms):
                col.plot(s_10[f][span2], '#6f83a3', label='wind_speed_10')  # plotting t, a separately
                col.plot(s_100[f][span2], '#9fabbf', label='wind_speed_100')  # plotting t, b separately
                col.plot(span2, t[f][span2], '#c91013', label='TARGETVAR_farm_' + str(f))  # plotting t, c separately
                col.plot(span1, p[f][span3], '#0707ad', label='PREDICTIONS_farm_' + str(f))  # plotting t, c separately
                col.legend(loc="upper right", prop={'size': 4})
            else:
                col.set_visible(False)
            f+=1

    fig.set_size_inches(24, 8)
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')

def plot_attention_scores(attn, path):
    fig, axs = plt.subplots(nrows=1, ncols=attn.shape[1])
    head = 0
    for ax in axs.reshape(-1):
        #scaler = sklearn.preprocessing.MinMaxScaler()
        #attn[0][0] = scaler.fit_transform(attn[0][head])
        plot = ax.imshow(attn[0][head], cmap='hot', interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(plot, cax)
        head += 1
        if(head == attn.shape[1]):
            break

    fig.set_size_inches(24, 8)
    fig.tight_layout(pad=2.0)
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')



