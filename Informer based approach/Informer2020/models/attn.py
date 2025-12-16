import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, My_ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 # the next args are optional, only needed if mask_flag = True
                 n_farms=None, n_cols=None, label_len=None, pred_len=None):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_farms = n_farms
        self.n_cols = n_cols
        self.label_len = label_len
        self.pred_len = pred_len

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only

            '''
                Here we have the decoder's sequence which should be masked.
                According to the Informer's definition of an insignificant query, a query is deemed
                insignificant if it's attention scores are all the same or about the same. In this case, the attention output
                for this query should be the mean of all tokens which are not hidden by the mask. 
                
                Our mask is as follows:
                All weather forecasts for all timestamps should be available.
                A timestamp corresponds to 50 tokens. The last 40 tokens are the weather forecasts.
                All these will be available and not hidden by the mask.
                The mask also hides the future timestamps (except for their weather forecasts) as usual.        
                
                The idea for calculating the means of tokens, which correspond to the assumption that the attention scores are all equal,
                is to use some kind of mask to hide the tokens we dont want to apply to the mean.
                Of course, the attention out for every 50 (supposedely insignificant) tokens (corresponding to a single timestamp), will be the same
                because the mask is applied based on entire timestamps and not based on the individual tokens.
                
                The mask will be of size (L_Q, L_Q, head_size) for all batches and number of heads.
                
                This mask will be contrasted with with a tensor of the same dimmentions we call V_exp, and the mean of a single line of V_exp
                after masking will be the attention out for a single (supposedely insignificant) query.
                
                Note that this is done for ll queries. Later in the code, the important queries will be replaced with their correct values.    
            '''

            # First expand the sequence. Note that eventually we will have to .clone in order for the backpropagation to work.
            # We have L_Q queries thus the shape will be the following.
            V_exp = V.unsqueeze(-3).expand(B, H, L_Q, L_Q, V.shape[-1])

            # Next we construct the mask. First fill the mask with all the same values.
            # Note that we also use .expand() in here because every n_cols lines will be the same.
            V_mask = torch.empty(list(V_exp.shape[-3:-1]),
                                 dtype=torch.bool
                                 ).fill_(True).to(V_exp.device).unsqueeze(-1).expand(
                list(V_exp.shape[-3:-1]) + [V.shape[-1]]).fill_(True).to(V_exp.device)

            n_timestamps_in_sequence = self.label_len + self.pred_len
            trig_l = torch.tril(torch.ones(n_timestamps_in_sequence, n_timestamps_in_sequence, dtype=torch.bool)).to(V_exp.device)



            # Hide the future timestamps
            V_mask = torch.einsum('ijklm,ij->ijklm', V_mask.reshape(
                n_timestamps_in_sequence,
                n_timestamps_in_sequence,
                self.n_cols,
                self.n_cols,
                V.shape[-1]), trig_l)



            V_mask = torch.einsum('ijkl->ikjl', V_mask.reshape(
                n_timestamps_in_sequence * n_timestamps_in_sequence,
                self.n_cols,
                V.shape[-1] * self.n_cols).reshape(
                n_timestamps_in_sequence,
                n_timestamps_in_sequence,
                self.n_cols,
                V.shape[-1] * self.n_cols)).reshape(
                L_Q, L_Q, V.shape[-1])

            # Reveal all weather forecasts for past and future timestamps.
            V_mask.reshape((-1, self.n_cols,V.shape[-1]))[:, self.n_farms:self.n_cols,:] = True

            # expand the mask for all heads and all batches
            V_mask = V_mask.unsqueeze(0).expand(H, L_Q, L_Q, V.shape[-1])
            V_mask = V_mask.unsqueeze(0).expand(B, H, L_Q, L_Q, V.shape[-1])

            # For convenience, we flip T-F
            V_mask = torch.logical_not(V_mask)

            # Fill with nan for positions we want to hide. Note we have to clone because we used expand previously
            V_exp = V_exp.clone().masked_fill_(V_mask, torch.nan)
            
            # Finally calculate the masked means
            # Note we can't use the word 'context' so we just use 'contex'
            contex = V_exp.nanmean(dim=-2)

            # This was the original version
            # contex = V.cumsum(dim=-2)

        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = My_ProbMask(B, H, L_Q, index, scores, device=V.device, n_farms=self.n_farms, n_cols=self.n_cols, label_len=self.label_len, pred_len=self.pred_len)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)



        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
