import argparse
import os
import torch

from exp.exp_informer import Exp_Informer
from exp.exp_informer_wind_farms import Exp_Informer_Wind_Farms
import numpy as np

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')



# Wind farms arguments:
parser.add_argument('--n_farms', type=int, default=10, help='number of farms in the dataset')
parser.add_argument('--n_cols', type=int, default=50, help='number of columns for every line in the dataset')
parser.add_argument('--n_weather_metrics_per_farm', type=int, default=4, help='number of weather prediction metrics for every farm in the dataset')
parser.add_argument('--n_farms_to_predict', type=int, default=10, help='number of farms we want the network to provide predictions. Note that if we want to predict for example the 1st farm only, then the first column of the dataset should be the farm we want to predict etc.')
parser.add_argument('--train_test_split', type=float, default=0.7, help='The percentage of the dataset to be allocated for training. The remaining will be the test set')
parser.add_argument('--test_set_month', type=int, default=-1, help='month to keep for test set. Default is -1 (last one). Use negative numbers.')
parser.add_argument('--n_preds_sliding_window', type=int, default=24, help='Total number of predictions for the sliding window method. Note this is NOT the same as pred_len which is the length of the predictions of the decoder layer.')
#-------------


parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')

parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    if s in ('True'):
        return True
    else:
        return False
    return s == True

parser.add_argument('--output_attention', type=boolean_string, default=False,  help='whether to output attention in ecoder')

parser.add_argument('--use_gpu', type=boolean_string, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=boolean_string, default=False, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1' ,help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    for id_ in device_ids:
        print(id_)
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer_Wind_Farms

setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des)

exp = Exp(args, setting) # set experiments
print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#exp.train(setting)

exp.train_sliding_windows(setting)

#_, _ = exp.create_plot_for_wind_farms_sliding_window(flag='test')

'''
losses_sum = 0
iters = 1000
loss = torch.nn.MSELoss()
for i in np.arange(iters):
    trues, preds = exp.create_plot_for_wind_farms_sliding_window(24)
    trues = torch.tensor(trues).detach()
    preds = torch.tensor(preds).detach()
    sw_loss = loss(preds, trues[args.seq_len:, :])
    losses_sum = losses_sum + sw_loss
    print(i+1)
    print(losses_sum/(i+1))
    torch.cuda.empty_cache()
print(losses_sum/iters)
'''


#trues, preds = exp.create_plot_for_wind_farms_sliding_window(24)
#print("Test the model ...")
#exp.test(setting)
    
torch.cuda.empty_cache()
