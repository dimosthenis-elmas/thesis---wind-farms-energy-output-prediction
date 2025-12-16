import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEmbedding_per_timestamp(nn.Module):
    def __init__(self, d_model, max_len=7000, n_farms=10, tokens_per_timestamp=50):
        super(PositionalEmbedding_per_timestamp, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        n_farms = n_farms
        n_weather_pred_cols = int((tokens_per_timestamp - n_farms)/n_farms)
        n_timestamps = math.ceil(max_len/(n_farms * (1+n_weather_pred_cols)))

        position = torch.tensor(np.arange(n_timestamps)).repeat_interleave(tokens_per_timestamp).repeat(n_timestamps).float().unsqueeze(1)
        position = position[:max_len]



        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    #original
                                    #kernel_size=3,
                                    kernel_size=1,
                                    #padding=padding,
                                    #padding_mode='circular'
                                   )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x


class TemporalEmbedding_coarse(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding_coarse, self).__init__()

        # In order to avoid encouraging overfitting we use a coarser  division of the day.
        # Instead of creating a different embedding for every hour we instead opt for a different embedding
        # every hour_size=4 (for example) hours. The same goes for the rest embeddings.
        # Note that this will have to change in case we are using a different dataset.
        # For example, for solar farms predictions the exact hour plays a far more important
        # role than for wind farms.

        self.hour_size = 4
        self.day_size = 8
        self.month_size = 12

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.hour_embed = Embed(24 // self.hour_size, d_model)
        self.day_embed = Embed(31 // self.day_size + 1, d_model)
        # + 1 because the month indices range from 1 to 12,
        # otherwise we would get an out of range error.
        self.month_embed = Embed(self.month_size + 1, d_model)

    def forward(self, x):
        x = x.long()
        hour_x = self.hour_embed(x[:, :, 3] // self.hour_size)
        day_x = self.day_embed((x[:, :, 1] - 1) // self.day_size)
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + day_x + month_x


class FarmIndexEmbedding(nn.Module):
    def __init__(self, d_model, n_farms, tokens_per_timestamp):
        super(FarmIndexEmbedding, self).__init__()
        self.tokens_per_timestamp = tokens_per_timestamp
        Embed = nn.Embedding
        self.farm_idx_embed = Embed(n_farms, d_model)
        n_weather_pred_cols = int((tokens_per_timestamp - n_farms)/n_farms)
        self.farm_id_stamp = torch.cat(
            (
                torch.arange(0, n_farms),
                torch.tensor(np.arange(n_farms)).repeat_interleave(n_weather_pred_cols)
            ), axis=0)

    def forward(self, x):
        x = x.long()
        n_timestamps = int(x.shape[-2]/self.farm_id_stamp.shape[-1])

        farm_indexes = self.farm_id_stamp.expand(n_timestamps, self.tokens_per_timestamp).reshape(-1,1)
        farm_indexes = farm_indexes.unsqueeze(0).expand([x.shape[0]] + list(farm_indexes.shape)).to(x.device)

        farm = self.farm_idx_embed(farm_indexes[:,:,0])

        return farm

class FeatureEmbedding(nn.Module):
    def __init__(self, d_model, tokens_per_timestamp):
        super(FeatureEmbedding, self).__init__()
        self.tokens_per_timestamp = tokens_per_timestamp
        self.d_model = d_model
        Embed = nn.Embedding
        self.feature_embed = Embed(self.tokens_per_timestamp, self.d_model)

        self.feature_id_stamp = torch.arange(0, self.tokens_per_timestamp)

    def forward(self, x):
        x = x.long()

        n_timestamps = x.reshape(x.shape[0], -1, self.tokens_per_timestamp).shape[1]

        feature_id_stamps = self.feature_id_stamp.expand(n_timestamps, self.tokens_per_timestamp).reshape(-1, 1)
        feature_id_stamps = feature_id_stamps.unsqueeze(0).expand([x.shape[0]] + list(feature_id_stamps.shape)).to(x.device)

        farm = self.feature_embed(feature_id_stamps[:,:,0])

        return farm


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, n_farms=10, tokens_per_timestamp=50):
        super(DataEmbedding, self).__init__()
        # since we want spatio temporal attention , we always provide tensors which have 1 in their last dimmention
        #This is both for the enc and the dec
        # So instead of 120,50 or 24,50 for each line of the dataset we have 120*50,1 or 24*50,1 etc.
        self.value_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.position_embedding = PositionalEmbedding_per_timestamp(d_model=d_model, n_farms=n_farms, tokens_per_timestamp=tokens_per_timestamp)
        #original
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.temporal_embedding = TemporalEmbedding_coarse(d_model=d_model, embed_type=embed_type, freq=freq)

        self.farm_idx_embedding = FarmIndexEmbedding(d_model=d_model, n_farms=n_farms, tokens_per_timestamp=tokens_per_timestamp)

        self.feature_embedding = FeatureEmbedding(d_model=d_model, tokens_per_timestamp = tokens_per_timestamp)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + self.farm_idx_embedding(x) + self.feature_embedding(x)
        
        return self.dropout(x)