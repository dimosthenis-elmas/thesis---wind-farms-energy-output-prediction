import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate, evaluate_v2
from dataset_forecasting_wind_farms import Dataset_Wind_Farms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
#parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

print(torch.__version__)

#torch.set_default_device('cuda:0')
print("-------------------------------")

pred_len = 24
context_len = 18


data_set = Dataset_Wind_Farms(seq_len=context_len, pred_len=pred_len,
            flag='train', scale=True, n_cols=50)

train_loader = DataLoader(
            data_set,
            batch_size=8,
            shuffle=True, generator=torch.Generator(device='cpu'), drop_last=True)

data_set = Dataset_Wind_Farms(seq_len=context_len, pred_len=pred_len,
            flag='test', scale=True, n_cols=50)

test_loader = DataLoader(
            data_set,
            batch_size=8,
            shuffle=True, generator=torch.Generator(device='cpu'), drop_last=True)

data_set = Dataset_Wind_Farms(seq_len=context_len, pred_len=pred_len,
            flag='validation', scale=True, n_cols=50)

valid_loader = DataLoader(
            data_set,
            batch_size=8,
            shuffle=True, generator=torch.Generator(device='cpu'), drop_last=True)

print("------------------------------------")

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == 'electricity':
    #target_dim = 370
    target_dim = 50

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

'''
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
)
'''

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)
'''
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

'''

model.load_state_dict(torch.load("./save/" + 'gefcom'
                                             '' + "/model.pth"))

evaluate_v2(
    model,
    test_loader,
    nsample=args.nsample)


