import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

# This is taken from the temporal fusion transformer
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation=None):

        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)

        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):

        if self.dropout_rate:
            x = self.dropout(x)

        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


# This is taken from the temporal fusion transformer
class GateAddNormNetwork(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation=None):
        super(GateAddNormNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.GLU = GatedLinearUnit(self.input_size,
                                   self.hidden_layer_size,
                                   self.dropout_rate,
                                   activation=self.activation_name)

        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)

        return output

# This is taken from the temporal fusion transformer
class GatedResidualNetwork(nn.Module):
    def __init__(self,
                 hidden_layer_size,
                 input_size=None,
                 output_size=None,
                 dropout_rate=None,
                 additional_context=None,
                 return_gate=False):

        super(GatedResidualNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate

        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)

        if self.additional_context:
            self.W3 = torch.nn.Linear(self.additional_context, self.hidden_layer_size, bias=False)

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.output_size,
                                                   self.dropout_rate)
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.hidden_layer_size,
                                                   self.dropout_rate)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):

        if self.additional_context:
            x, context = x
            # x_forward = self.W2(x)
            # context_forward = self.W3(context)
            # print(self.W3(context).shape)
            n2 = F.elu(self.W2(x) + self.W3(context))
        else:
            n2 = F.elu(self.W2(x))

        # print('n2 shape {}'.format(n2.shape))

        n1 = self.W1(n2)

        # print('n1 shape {}'.format(n1.shape))

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        # print('output shape {}'.format(output.shape))

        return output

# This is taken from the temporal fusion transformer
class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_layer_size,
                 dropout_rate,
                 output_size,
                 input_size=None,
                 additional_context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                  input_size=self.input_size,
                                                  output_size=self.output_size,
                                                  dropout_rate=self.dropout_rate,
                                                  additional_context=self.additional_context)

        self.per_feature_grn = nn.ModuleList([GatedResidualNetwork(self.hidden_layer_size,
                                                                   dropout_rate=self.dropout_rate)
                                              for i in range(self.output_size)])

    def forward(self, x):
        # Non Static Inputs
        if self.additional_context:
            embedding, static_context = x
            # print('static_context')
            # print(static_context.shape)

            time_steps = embedding.shape[1]
            flatten = embedding.view(-1, time_steps, self.hidden_layer_size * self.output_size)
            # print('flatten')
            # print(flatten.shape)

            static_context = static_context.unsqueeze(1)
            # print('static_context')
            # print(static_context.shape)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = self.flattened_grn((flatten, static_context))
            # print('mlp_outputs')
            # print(mlp_outputs.shape)

            sparse_weights = F.softmax(mlp_outputs, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(2)
            # print('sparse_weights')
            # print(sparse_weights.shape)

            trans_emb_list = []
            for i in range(self.output_size):
                e = self.per_feature_grn[i](embedding[Ellipsis, i])
                trans_emb_list.append(e)
            transformed_embedding = torch.stack(trans_emb_list, axis=-1)
            # print('transformed_embedding')
            # print(transformed_embedding.shape)

            combined = sparse_weights * transformed_embedding
            # print('combined')
            # print(combined.shape)

            temporal_ctx = torch.sum(combined, dim=-1)
            # print('temporal_ctx')
            # print(temporal_ctx.shape)

        # Static Inputs
        else:
            embedding = x
            # print('embedding')
            # print(embedding.shape)

            flatten = torch.flatten(embedding, start_dim=2)
            # flatten = embedding.view(batch_size, -1)
            # print('flatten')
            # print(flatten.shape)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = self.flattened_grn(flatten)
            # print('mlp_outputs')
            # print(mlp_outputs.shape)

            #sparse_weights = F.softmax(mlp_outputs, dim=-1)
            sparse_weights = F.tanh(mlp_outputs)
            sparse_weights = sparse_weights.unsqueeze(2)
            #             print('sparse_weights')
            #             print(sparse_weights.shape)

            trans_emb_list = []
            for i in range(self.output_size):
                # print('embedding for the per feature static grn')
                # print(embedding[:, i:i + 1, :].shape)
                e = self.per_feature_grn[i](embedding[Ellipsis, i])
                trans_emb_list.append(e)
            transformed_embedding = torch.stack(trans_emb_list, axis=-1)
            #             print('transformed_embedding')
            #             print(transformed_embedding.shape)

            combined = sparse_weights * transformed_embedding
            #             print('combined')
            #             print(combined.shape)

            temporal_ctx = combined
            #temporal_ctx = torch.sum(combined, dim=-1)
        #             print('temporal_ctx')
        #             print(temporal_ctx.shape)

        return temporal_ctx, sparse_weights


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

        self.variable_selection_layers_historical_context = nn.ModuleList(
            [
                VariableSelectionNetwork(
                    hidden_layer_size=64,
                    dropout_rate=0.1,
                    output_size=50,
                    input_size=64 * 50
                )
                for _ in range(config["layers"])
            ]
        )

        self.variable_selection_layers_future = nn.ModuleList(
            [
                VariableSelectionNetwork(
                    hidden_layer_size=64,
                    dropout_rate=0.1,
                    output_size=50,
                    input_size=64 * 50
                )
                for _ in range(config["layers"])
            ]
        )

        self.historical_lstm_layers = nn.ModuleList(
            [
                nn.LSTM(input_size=64 * 50,
                        hidden_size=64 * 50,
                        batch_first=True)
                for _ in range(config["layers"])
            ]
        )

        self.future_lstm_layers = nn.ModuleList(
            [
                nn.LSTM(input_size=64 * 50,
                        hidden_size=64 * 50,
                        batch_first=True)
                for _ in range(config["layers"])
            ]
        )

        self.post_seq_encoder_gate_add_norm_layers = nn.ModuleList(
            [
                GateAddNormNetwork(64 * 50,
                                   64 * 50,
                                   0.1,
                                   activation=None)
                for _ in range(config["layers"])
            ]
        )
        #-------------------------------------------


        # todo: replace these hardcoded arguments with proper parameters
        self.variable_selection_layer_historical_context = VariableSelectionNetwork(
            hidden_layer_size=64,
            dropout_rate=0.1,
            output_size=50,
            input_size=64*50
        )

        # todo: replace these hardcoded arguments with proper parameters
        self.variable_selection_layer_future = VariableSelectionNetwork(
            hidden_layer_size=64,
            dropout_rate=0.1,
            output_size=50,
            input_size=64 * 50
        )

        # todo: replace these hardcoded arguments with proper parameters
        self.historical_lstm = nn.LSTM(input_size=64*50,
                                       hidden_size=64*50,
                                       batch_first=True)

        # todo: replace these hardcoded arguments with proper parameters
        self.future_lstm = nn.LSTM(input_size=64*50,
                                   hidden_size=64*50,
                                   batch_first=True)

        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(64*50,
                                                                 64*50,
                                                                 0.1,
                                                                 activation=None)
        #-------------------------------------

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

    '''
    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []

        # todo: replace these hardcoded arguments with proper parameters
        context_len = 18
        pred_len = 24

        for index, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond_info, diffusion_emb)

            historical_features, _ = self.variable_selection_layers_historical_context[index](x[:,:,:,:context_len].permute(0,3,1,2))
            historical_features = historical_features.permute(0,2,3,1)

            future_features, _ = self.variable_selection_layers_future[index](x[:, :, :, -pred_len:].permute(0, 3, 1, 2))
            future_features = future_features.permute(0, 2, 3, 1)

            input_embeddings = torch.cat((historical_features, future_features), -1)


            history_lstm, (state_h, state_c) = self.historical_lstm_layers[index](historical_features.permute(0,3,1,2).view(*historical_features.permute(0,3,1,2).size()[:-2],-1))

            future_lstm, _ = self.future_lstm_layers[index](future_features.permute(0,3,1,2).view(*future_features.permute(0,3,1,2).size()[:-2],-1),
                                              (state_h,
                                               state_c))

            #   todo remove hardcoded stuff
            lstm_layer = torch.cat((history_lstm, future_lstm), axis=1).view(8,(context_len + pred_len),64*50)

            residual_after_variable_selection = self.post_seq_encoder_gate_add_norm_layers[index](lstm_layer, input_embeddings.permute(0,3,1,2).view(8,(context_len + pred_len),64*50))
            residual_after_variable_selection = residual_after_variable_selection.view(*residual_after_variable_selection.shape[:2], 64, 50).permute(0, 2, 3, 1)

            x = residual_after_variable_selection


            x = input_embeddings

            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x
        '''


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
