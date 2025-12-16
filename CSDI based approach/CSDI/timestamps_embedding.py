import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TimestampsEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimestampsEmbedding, self).__init__()

        # In order to avoid encouraging overfitting we use a coarser  division of the day.
        # Instead of creating a different embedding for every hour we instead opt for a different embedding
        # every hour_size=4 (for example) hours. The same goes for the rest embeddings.
        # Note that this will have to change in case we are using a different dataset.
        # For example, for solar farms predictions the exact hour plays a far more important
        # role than for wind farms.

        self.hour_size = 4
        self.day_size = 8
        self.month_size = 12

        Embed = nn.Embedding

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