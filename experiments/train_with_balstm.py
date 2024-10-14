import logging
import os
import argparse
import pandas as pd
import importlib
import numpy as np
import torch
import datetime
from torchhydro.configs.config import cmd,default_config_file, update_cfg
from torchhydro.datasets.data_balstm import WithStatic, TrainTestSplit
from torchhydro.models.balstm_model import BALSTM,inverse_normalize
from torchhydro.models.crits import NMSELoss
from torchhydro.trainers.train_balstm import BALSTMTrainer
from torchhydro.trainers.trainer import train_and_evaluate, set_random_seed
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.metric import SingleNSE,seqKGE

def train_balstm(config):
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
    epochs = config['training_cfgs']['epochs']
    log_step = config['training_cfgs']['log_step']
    milestones = config['training_cfgs']['milestones']
    gamma = config['training_cfgs']['gamma']
    learning_rate = config['training_cfgs']['lr_scheduler']['lr']
    seed=config['training_cfgs']['random_seed']
    
    metrics_fqns = config['model_cfgs']['model_hyperparam']['metrics_fqns']
    batch_size = config['model_cfgs']['model_hyperparam']['batch_size']
    hidden_size = config['model_cfgs']['model_hyperparam']['hidden_size']
    num_layers = config['model_cfgs']['model_hyperparam']['num_layers']
    dropout = config['model_cfgs']['model_hyperparam']['dropout']
    
    # 创建数据集
    dataset = WithStatic(gages_dir, gages_dir1, S_dir, xd_dir, xg_dir, y_dir, seq_length, 
                         input_size_dyn, input_size_glo, input_size_sta, output_size)
    # 数据集分割
    datasplit = TrainTestSplit(dataset, batch_size, num_workers=num_workers, pin_memory=False, test_size=test_size)
    # 设置随机种子
    set_random_seed(seed)
    # 训练配置
    n, accumulation = 0, 0
    for train_loader, test_loader in datasplit:
        model = BALSTM(input_size_sta, input_size_dyn, input_size_glo, hidden_size,
                       output_size, num_layers, dropout)
        criterion = NMSELoss(dataset)
        metrics_fqns = metrics_fqns
        metrics = []
        for fqn in metrics_fqns:
            module_name, class_name = fqn.rsplit(".", 1)
            MetricClass = getattr(importlib.import_module(module_name), class_name)
            metrics.append(MetricClass())
        metrics = [met for met in metrics]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        trainer = BALSTMTrainer(model, criterion, metrics, optimizer, n_gpu, epochs, log_step, 
                                data_loader=train_loader, valid_data_loader=test_loader, 
                                lr_scheduler=lr_scheduler, round=n)
        accumulation += trainer.train()
        n += 1
    print(accumulation / n)
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
            "n_gpu": 1,#ctx
            "epochs": 3,
            "log_step": 1000,
            "milestones": [700],
            "gamma": 0.5,
            "learning_rate": 0.0001,
            "seed":2,#rs
            "train_mode": True,
        },
        evaluation_cfgs={
            "evaluate_model_name":"balstm/model_latest.pth"
        }
    )

    update_cfg(config_data, args)
    return config_data

if __name__ == "__main__":
    config=create_config_BALSTM()
    if config['training_cfgs']['train_mode']:
        print("Training with"," epochs",config['training_cfgs']['epochs'])
        train_balstm(config)
    else:
        print("Evaluating with ",config['evaluation_cfgs']['evaluate_model_name'])
        evaluate_balstm(config)