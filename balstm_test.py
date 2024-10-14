import importlib
import random
import numpy as np
import pandas as pd
import torch
from torchhydro.datasets.data_balstm import WithStatic, TrainTestSplit
from torchhydro.models.balstm_model import BALSTM
from torchhydro.models.crits import NMSELoss
from torchhydro.trainers.train_balstm import BALSTMTrainer
from torchhydro.trainers.trainer import *
S_dir = 'data/balstm/grdc_attributes1.csv'
gages_dir = 'data/balstm/basin_list.xlsx'
gages_dir1 = 'data/balstm/basin_def.xlsx'
y_dir = 'data/balstm/runoff_data/txt'
xd_dir = 'data/balstm/forcing_t'
xg_dir = 'data/balstm/global_data/output'
seq_length = 12
input_size_dyn = 13
input_size_glo = 106
output_size = 1
input_size_sta = 195

dataset = WithStatic(gages_dir, gages_dir1, S_dir, xd_dir, xg_dir, y_dir, seq_length, 
                     input_size_dyn, input_size_glo, input_size_sta, output_size)


datasplit = TrainTestSplit(dataset, batch_size=32, num_workers=0, pin_memory=False, test_size=0.2)

set_random_seed(42)
n_gpu = 1
epochs = 100
log_step = 1000
milestones = [700]
gamma = 0.5
learning_rate = 0.0001
n, accumulation = 0, 0

for train_loader, test_loader in datasplit:
    model = BALSTM(input_size_sta, input_size_dyn, input_size_glo, hidden_size=64,
                   output_size=1, num_layers=2, drop_prob=0.4)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    criterion = NMSELoss(dataset)
    metrics_fqns = [
        "trainer.metric.NSE",
        "trainer.metric.MeanNSE",
        "trainer.metric.MedianNSE",
    ]

    metrics = []
    for fqn in metrics_fqns:
        module_name, class_name = fqn.rsplit(".", 1)
        MetricClass = getattr(importlib.import_module(module_name), class_name)
        metrics.append(MetricClass())
    metrics = [met for met in metrics]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    trainer = BALSTMTrainer(model, criterion, metrics, optimizer, n_gpu, epochs, log_step, 
                            data_loader=train_loader, valid_data_loader=test_loader, 
                            lr_scheduler=lr_scheduler, round=n)
    accumulation += trainer.train()
    n += 1

print(accumulation / n)