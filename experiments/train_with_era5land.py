"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-27 08:43:55
LastEditors: Wenyu Ouyang
Description: 
FilePath: \torchhydro\experiments\train_with_era5land.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os
import argparse
import pandas as pd
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv("data/basin_615.csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Train a Seq2Seq or Transformer model."
    )
    parser.add_argument(
        "--m_name",
        type=str,
        choices=["Seq2Seq", "Transformer"],
        default="Seq2Seq",
        help="Type of model to train: Seq2Seq or Transformer",
    )

    args = parser.parse_args()

    if args.m_name == "Seq2Seq":
        config_data = create_config_Seq2Seq()
    elif args.m_name == "Transformer":
        config_data = create_config_Transformer()

    train_and_evaluate(config_data)


def create_config_Seq2Seq():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("train_with_ear5land", "ex1_539_basins")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[2],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 20,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 56,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=256,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2016-01-01-01", "2023-11-30-01")],
        test_period=[("2015-01-01-01", "2016-01-01-01")],
        valid_period=[("2015-01-01-01", "2016-01-01-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="batch",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        model_type="MTL",
        # continue_train=True,
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


def create_config_Transformer():
    project_name = os.path.join("train_with_era5land_trans", "ex1")
    config_data = default_config_file()

    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Transformer",
        model_hyperparam={
            "n_encoder_inputs": 20,
            "n_decoder_inputs": 19,
            "n_decoder_output": 2,
            "channels": 256,
            "num_embeddings": 512,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "prec_window": 0,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        batch_size=128,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="TransformerDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2015-05-01-01", "2022-12-20-01")],
        test_period=[("2023-01-01-01", "2023-11-20-01")],
        valid_period=[("2023-01-01-01", "2023-11-20-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="sequence",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        patience=10,
        model_type="MTL",
    )
    update_cfg(config_data, args)
    return config_data


if __name__ == "__main__":
    main()
