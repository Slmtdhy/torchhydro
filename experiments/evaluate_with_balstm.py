import datetime
import logging
import os
import argparse
import pandas as pd
import importlib
import numpy as np
import torch
from torchhydro.datasets.data_balstm import WithStatic, TrainTestSplit
from models.balstm_model import BALSTM
from torchhydro.trainers.metric import SingleNSE,seqKGE
from torchhydro.configs.config import cmd, default_config_file, update_cfg
# Assuming you have the normalize function defined
def inverse_normalize(data, mean, std):
    return data * std + mean
def create_config_BALSTM():
    # 设置项目名称和默认配置文件路径
    project_name = os.path.join("train_with_BALSTM", "ex1")
    config_data = default_config_file()
    # 填充命令行参数到配置中
    args = cmd(
        sub=project_name,
        # source="BALSTM",
        # batch_size=64,
        model_hyperparam= {
                "batch_size": 32,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.4,
                "metrics_fqns": [
                "trainers.metric.NSE",
                "trainers.metric.MeanNSE",
                "trainers.metric.MedianNSE",
                ],
            },
        data_cfgs={
            "S_dir": "data/balstm/grdc_attributes1.csv",
            "gages_dir": "data/balstm/basin_list.xlsx",
            "gages_dir1": "data/balstm/basin_def.xlsx",
            "y_dir": "data/balstm/runoff_data/txt",
            "xd_dir": "data/balstm/forcing_t",
            "xg_dir": "data/balstm/global_data/output",
            "seq_length": 12,
            "input_size_dyn": 13,
            "input_size_glo": 106,
            "input_size_sta": 195,
            "output_size": 1,
            "test_size": 0.2,
            "num_workers": 0,
        },
        training_cfgs= {
            "n_gpu": 1,
            "epochs": 3,
            "log_step": 1000,
            "milestones": [700],
            "gamma": 0.5,
            "learning_rate": 0.0001,
            "seed":2,
            "train_mode": True,
        },
        evaluation_cfgs={
            "evaluate_model_name":"balstm/checkpoint-epoch97.pth"
        }
    )

    update_cfg(config_data, args)
    return config_data
def evaluate_balstm(config):
    # 数据路径配置
    S_dir = config['data_cfgs']['S_dir']
    gages_dir = config['data_cfgs']['gages_dir']
    gages_dir1 = config['data_cfgs']['gages_dir1']
    y_dir = config['data_cfgs']['y_dir']
    xd_dir = config['data_cfgs']['xd_dir']
    xg_dir = config['data_cfgs']['xg_dir']
    seq_length = config['data_cfgs']['seq_length']
    input_size_dyn = config['data_cfgs']['input_size_dyn']
    input_size_glo = config['data_cfgs']['input_size_glo']
    input_size_sta = config['data_cfgs']['input_size_sta']
    output_size = config['data_cfgs']['output_size']
    test_size=config['data_cfgs']['test_size']
    num_workers=config['data_cfgs']['num_workers']

    n_gpu = config['training_cfgs']['device']
    
    batch_size = config['model_cfgs']['model_hyperparam']['batch_size']
    hidden_size = config['model_cfgs']['model_hyperparam']['hidden_size']
    num_layers = config['model_cfgs']['model_hyperparam']['num_layers']
    dropout = config['model_cfgs']['model_hyperparam']['dropout']

    model_name=config['evaluation_cfgs']['evaluate_model_name']
    # 创建数据集
    dataset = WithStatic(gages_dir, gages_dir1, S_dir, xd_dir, xg_dir, y_dir, seq_length, 
                         input_size_dyn, input_size_glo, input_size_sta, output_size)
    # 数据集分割
    datasplit = TrainTestSplit(dataset, batch_size, num_workers=num_workers, pin_memory=False, test_size=test_size)
    model = BALSTM(input_size_sta, input_size_dyn, input_size_glo, hidden_size,output_size, num_layers, dropout)
    _, _, _, _, _, _, y_mean, y_std = dataset.norm
    pth = torch.load(model_name)
    test_loader = next(iter(datasplit))[1]
    # load trained weights
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(pth['state_dict'])
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # simulate
    with torch.no_grad():
        outputs = []
        targets = []
        labels = []
        times = []
        for input, target, label, time in test_loader:
            target = target.to(device)
            if isinstance(input, torch.Tensor):
                input = input.to(device)
            else:
                input = tuple(i.to(device) for i in input)
            outputs.append(model(input))
            # print(outputs)
            targets.append(target)
            labels.append(label)
            times.append(time)
        # concatenate in dim 0
        y_pred = torch.cat(outputs)
        y_true = torch.cat(targets)
        basin = torch.cat(labels)
        Time = torch.cat(times)

    mask = ~torch.isnan(y_true)  # Create a mask to exclude NaN values
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    mask = mask.flatten().cpu()
    basin = basin[mask]
    Time = Time[mask]
    # 反归一化 y_true and y_pred
    y_true_denorm = inverse_normalize(y_true.cpu().numpy(), y_mean, y_std)
    y_pred_denorm = inverse_normalize(y_pred.cpu().numpy(), y_mean, y_std)
    # 创建一个向量化函数
    fromordinal_vectorized = np.vectorize(lambda ordinal: datetime.datetime.fromordinal(ordinal).date())
    # 使用向量化函数转换时间
    Time = fromordinal_vectorized(Time)

    nse_calculator = SingleNSE()
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    df['STAID'] = dataset.gages['STAID']
    df['KGE'] = seqKGE(y_pred, y_true, basin).cpu().numpy()
    df['NSE'] = nse_calculator(y_pred, y_true, basin).cpu().numpy()
    df1['basin'] = basin.cpu().numpy().ravel()
    df1['y_true'] = y_true_denorm.ravel()
    df1['y_pred'] = y_pred_denorm.ravel()
    df1['time'] = Time.ravel()
    df.to_csv('basin-results6.csv', index=None)
    df1.to_csv('basin-data6.csv', index=None)

if __name__ == '__main__':
    config=create_config_BALSTM()
    evaluate_balstm(config)