import os
import numpy as np
import pytest

import torch

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.models.kernel_conv import uh_conv, uh_gamma


@pytest.fixture()
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device):
    dpl_ = DplLstmXaj(5, 15, 64, kernel_size=15, warmup_length=5)
    return dpl_.to(device)


@pytest.fixture()
def dpl_eh(device):
    dpl_ = DplLstmXaj(
        5,
        15,
        64,
        kernel_size=15,
        warmup_length=5,
        source_book="EH",
        source_type="sources",
    )
    return dpl_.to(device)


@pytest.fixture()
def dpl_eh5mm(device):
    dpl_ = DplLstmXaj(
        5,
        15,
        64,
        kernel_size=15,
        warmup_length=5,
        source_book="EH",
        source_type="sources5mm",
    )
    return dpl_.to(device)


def test_dpl_lstm_xaj(device, dpl):
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    x = torch.rand(20, 10, 5).to(device)
    z = torch.rand(20, 10, 5).to(device)
    q = dpl(x, z)
    assert len(q.shape) == 3
    assert q.shape == (15, 10, 1)
    q.backward(torch.ones_like(q))
    assert type(q) == torch.Tensor


def test_dpl_lstm_xaj_eh(device, dpl_eh):
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    x = torch.rand(20, 10, 5).to(device)
    z = torch.rand(20, 10, 5).to(device)
    q = dpl_eh(x, z)
    assert len(q.shape) == 3
    assert q.shape == (15, 10, 1)
    q.backward(torch.ones_like(q))
    assert type(q) == torch.Tensor


def test_dpl_lstm_xaj_eh5mm(device, dpl_eh5mm):
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    x = torch.rand(20, 10, 5).to(device)
    z = torch.rand(20, 10, 5).to(device)
    q = dpl_eh5mm(x, z)
    assert len(q.shape) == 3
    assert q.shape == (15, 10, 1)
    q.backward(torch.ones_like(q))
    assert type(q) == torch.Tensor


def test_uh_gamma():
    # batch = 10
    tempa = torch.Tensor(np.full(10, [2.5]))
    tempb = torch.Tensor(np.full(10, [3.5]))
    # repeat for 20 periods and add one dim as feature: time_seq, batch, feature
    routa = tempa.repeat(20, 1).unsqueeze(-1)
    routb = tempb.repeat(20, 1).unsqueeze(-1)
    uh = uh_gamma(routa, routb, len_uh=15)
    np.testing.assert_almost_equal(
        uh.numpy()[:, 0, :],
        np.array(
            [
                [0.0069],
                [0.0314],
                [0.0553],
                [0.0738],
                [0.0860],
                [0.0923],
                [0.0939],
                [0.0919],
                [0.0875],
                [0.0814],
                [0.0744],
                [0.0670],
                [0.0597],
                [0.0525],
                [0.0459],
            ]
        ),
        decimal=3,
    )


def test_uh():
    uh_from_gamma = torch.full((5, 3, 1), 1.0)
    # uh_from_gamma = torch.Tensor(np.arange(15).reshape(5, 3, 1))
    rf = torch.Tensor(np.arange(30).reshape(10, 3, 1) / 100)
    qs = uh_conv(rf, uh_from_gamma)
    np.testing.assert_almost_equal(
        np.array(
            [
                [0.0000, 0.0100, 0.0200],
                [0.0300, 0.0500, 0.0700],
                [0.0900, 0.1200, 0.1500],
                [0.1800, 0.2200, 0.2600],
                [0.3000, 0.3500, 0.4000],
                [0.4500, 0.5000, 0.5500],
                [0.6000, 0.6500, 0.7000],
                [0.7500, 0.8000, 0.8500],
                [0.9000, 0.9500, 1.0000],
                [1.0500, 1.1000, 1.1500],
            ]
        ),
        qs.numpy()[:, :, 0],
        decimal=3,
    )


def test_train_evaluate_dpl(dpl_args, config_data):
    update_cfg(config_data, dpl_args)
    train_and_evaluate(config_data)


@pytest.fixture()
def dpl_selfmadehydrodataset_args():
    project_name = os.path.join("test_camels", "expdpl002")
    train_period = ["1985-10-01", "1986-04-01"]
    # valid_period = ["1995-10-01", "2000-10-01"]
    valid_period = None
    test_period = ["2000-10-01", "2001-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": "/mnt/c/Users/wenyu/OneDrive/Research/research_topic_advancement/research_progress_plan/data4dpl/dplARdata",
        },
        ctx=[0],
        model_name="DplLstmXaj",
        # model_name="DplAttrXaj",
        model_hyperparam={
            "n_input_features": 25,
            # "n_input_features": 19,
            "n_output_features": 15,
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 30,
            "param_limit_func": "clamp",
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_sum",
                "potential_evaporation_sum",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=50,
        forecast_history=0,
        forecast_length=60,
        var_t=[
            "total_precipitation_sum",
            "potential_evaporation_sum",
            "snow_depth_water_equivalent",
            "surface_net_solar_radiation_sum",
            "surface_pressure",
            "temperature_2m",
        ],
        var_c=[
            "sgr_dk_sav",
            "pet_mm_syr",
            "slp_dg_sav",
            "for_pc_sse",
            "pre_mm_syr",
            "slt_pc_sav",
            "swc_pc_syr",
            "soc_th_sav",
            "cly_pc_sav",
            "ari_ix_sav",
            "snd_pc_sav",
            "ele_mt_sav",
            "area",
            "tmp_dc_syr",
            "crp_pc_sse",
            "lit_cl_smj",
            "wet_cl_smj",
            "snw_pc_syr",
            "glc_cl_smj",
        ],
        var_out=["streamflow"],
        target_as_input=0,
        # constant_only=1,
        train_epoch=2,
        model_loader={
            "load_way": "specified",
            "test_epoch": 2,
        },
        warmup_length=30,
        opt="Adadelta",
        which_first_tensor="sequence",
    )


def test_dpl_selfmadehydrodataset(dpl_selfmadehydrodataset_args):
    cfg = default_config_file()
    update_cfg(cfg, dpl_selfmadehydrodataset_args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
