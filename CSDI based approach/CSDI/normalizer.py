import torch

class Normalizer(torch.nn.Module):
    def __init__(self, n_feats):
        # The normalizer returns a signal with zero mean and unit variance (z-scores).
        super(Normalizer, self).__init__()

        self.batch_norm = torch.nn.BatchNorm2d(n_feats, affine=False, track_running_stats=True)


    def forward(self, x):
        # Shape of x should be like this: (batch_size, n_feats, n_hours, 1)
        # returns shape same as the input
        return self.batch_norm(x)

    def denormalize(self, x):
        # Shape of x should be like this: (batch_size, n_feats, n_hours, 1)
        # returns shape same as the input
        x = self.batch_norm.running_mean + x.transpose(1, 3) * self.batch_norm.running_var ** 0.5
        return x.transpose(1, 3)