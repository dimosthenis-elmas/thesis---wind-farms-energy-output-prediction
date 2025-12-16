import torch
import numpy

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


# This is a variation of the standard informer mask.
# We use this mask in order to allow the decoder to always attend to all
# the weather predictions related to future timesteps
class My_ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu", n_farms = 10, n_cols = 50, label_len=48, pred_len=24):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # If the query's position is not at the end of the n_cols columns, then do not apply the mask for all n_cols.
        # This is because every n_cols tokens (keys) are associated with a single timestamp (row of the dataset).
        index_timestamp_based = torch.ceil(index/n_cols)*n_cols - 1
        index_timestamp_based = index_timestamp_based.type(dtype=torch.int)
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index_timestamp_based, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

        mask2 = torch.empty(list(self._mask.shape[:-1]) + [(label_len + pred_len) * n_cols], dtype=torch.bool).fill_(False).to(device)
        mask2.reshape((-1, n_cols))[:, n_farms:n_cols] = True
        mask2.flatten()
        self._mask = self._mask.masked_fill_(mask2, False)
    @property
    def mask(self):
        return self._mask