"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2024-08-10 15:10:27
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import math
import re
import sys
from datetime import datetime
from datetime import timedelta
from itertools import chain
from typing import Optional

import geopandas as gpd
import hydrotopo.ig_path as htip
import numpy as np
import pandas as pd
import torch
import xarray as xr
from dateutil.parser import parse
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.utils.utils import streamflow_unit_conv
from torch.utils.data import Dataset
from tqdm import tqdm

from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict,
)

LOGGER = logging.getLogger(__name__)


def _fill_gaps_da(da: xr.DataArray, fill_nan: Optional[str] = None) -> xr.DataArray:
    """Fill gaps in a DataArray"""
    if fill_nan is None or da is None:
        return da
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        all_non_nan_idx = []
        for i in range(da.shape[0]):
            non_nan_idx_tmp = np.where(~np.isnan(da[i].values))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        for i in range(da.shape[0]):
            targ_i = da[i][non_nan_idx]
            da[i][non_nan_idx] = targ_i.interpolate_na(
                dim="time", fill_value="extrapolate"
            )
    elif fill_nan == "mean":
        # fill with mean
        for var in da["variable"].values:
            var_data = da.sel(variable=var)  # select the data for the current variable
            mean_val = var_data.mean(
                dim="basin"
            )  # calculate the mean across all basins
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[dict(variable=var)] = (
                filled_data  # update the original dataarray with the filled data
            )
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="time", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


def detect_date_format(date_str):
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            parameters for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    @property
    def data_source(self):
        source_name = self.data_cfgs["source_cfgs"]["source_name"]
        source_path = self.data_cfgs["source_cfgs"]["source_path"]
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        return data_sources_dict[source_name](source_path, **other_settings)

    @property
    def streamflow_name(self):
        return self.data_cfgs["target_cols"][0]

    @property
    def precipitation_name(self):
        return self.data_cfgs["relevant_cols"][0]

    @property
    def ngrid(self):
        """How many basins/grids in the dataset

        Returns
        -------
        int
            number of basins/grids
        """
        return len(self.basins)

    @property
    def nt(self):
        """length of longest time series in all basins

        Returns
        -------
        int
            number of longest time steps
        """
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            earliest_date = None
            latest_date = None
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                date_format = detect_date_format(start_date_str)

                start_date = datetime.strptime(start_date_str, date_format)
                end_date = datetime.strptime(end_date_str, date_format)

                if earliest_date is None or start_date < earliest_date:
                    earliest_date = start_date
                if latest_date is None or end_date > latest_date:
                    latest_date = end_date
            earliest_date = earliest_date.strftime(date_format)
            latest_date = latest_date.strftime(date_format)
        else:
            trange_type_num = 1
            earliest_date = self.t_s_dict["t_final_range"][0]
            latest_date = self.t_s_dict["t_final_range"][1]
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        s_date = pd.to_datetime(earliest_date)
        e_date = pd.to_datetime(latest_date)
        time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return len(time_series)

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the times of all basins

        TODO: Although we support get different time ranges for different basins,
        we didn't implement the reading function for this case in _read_xyc method.
        Hence, it's better to choose unified time range for all basins
        """
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            times_ = []
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            detect_date_format(self.t_s_dict["t_final_range"][0][0])
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                s_date = pd.to_datetime(start_date_str)
                e_date = pd.to_datetime(end_date_str)
                time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
                times_.append(time_series)
        else:
            detect_date_format(self.t_s_dict["t_final_range"][0])
            trange_type_num = 1
            s_date = pd.to_datetime(self.t_s_dict["t_final_range"][0])
            e_date = pd.to_datetime(self.t_s_dict["t_final_range"][1])
            times_ = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return times_

    def __len__(self):
        return self.num_samples if self.train_mode else self.ngrid

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x[item, :, :]
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length: idx + self.rho + self.horizon, :]
        y = self.y[basin, idx: idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _pre_load_data(self):
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.horizon = self.data_cfgs["forecast_length"]

    def _load_data(self):
        self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()

    def _normalize(self):
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c

    def _to_dataarray_with_unit(self, data_forcing_ds, data_output_ds, data_attr_ds):
        # trans to dataarray to better use xbatch
        if data_output_ds is not None:
            data_output = self._trans2da_and_setunits(data_output_ds)
        else:
            data_output = None
        if data_forcing_ds is not None:
            data_forcing = self._trans2da_and_setunits(data_forcing_ds)
        else:
            data_forcing = None
        if data_attr_ds is not None:
            # firstly, we should transform some str type data to float type
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None
        return data_forcing, data_output, data_attr

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : _type_
            _description_
        data_output_ds : _type_
            _description_
        """

        def standardize_unit(unit):
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        streamflow_unit = data_output_ds[self.streamflow_name].attrs["units"]
        prcp_unit = data_forcing_ds[self.precipitation_name].attrs["units"]

        standardized_streamflow_unit = standardize_unit(streamflow_unit)
        standardized_prcp_unit = standardize_unit(prcp_unit)
        if standardized_streamflow_unit != standardized_prcp_unit:
            data_output_ds = streamflow_unit_conv(
                data_output_ds,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
        return data_forcing_ds, data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        # x
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
        )

    def _trans2da_and_setunits(self, ds):
        """Set units for dataarray transfromed from dataset"""
        result = ds.to_array(dim="variable")
        units_dict = {
            var: ds[var].attrs["units"]
            for var in ds.variables
            if "units" in ds[var].attrs
        }
        result.attrs["units"] = units_dict
        return result

    def _kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            _fill_gaps_da(x, fill_nan="interpolate")
            warn_if_nan(x)
        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)
        if c_rm_nan:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        warn_if_nan(x, nan_mode="all")
        warn_if_nan(y, nan_mode="all")
        warn_if_nan(c, nan_mode="all")
        return x, y, c

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        max_time_length = self.nt
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if self.is_tra_val_te != "train":
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho: f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(BasinSingleFlowDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y

    def __len__(self):
        return self.num_samples


class DplDataset(BaseDataset):
    """pytorch dataset for Differential parameter learning"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            configs for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(DplDataset, self).__init__(data_cfgs, is_tra_val_te)
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # For physical hydrological models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = data_cfgs["warmup_length"]
        self.target_as_input = data_cfgs["target_as_input"]
        self.constant_only = data_cfgs["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataset(data_cfgs, is_tra_val_te="train")

    def __getitem__(self, item):
        """
        Get one mini-batch for dPL (differential parameter learning) model

        TODO: not check target_as_input and constant_only cases yet

        Parameters
        ----------
        item
            index

        Returns
        -------
        tuple
            a mini-batch data;
            x_train (not normalized forcing), z_train (normalized data for DL model), y_train (not normalized output)
        """
        warmup = self.warmup_length
        rho = self.rho
        horizon = self.horizon
        if self.train_mode:
            xc_norm, _ = super(DplDataset, self).__getitem__(item)
            basin, time = self.lookup_table[item]
            if self.target_as_input:
                # y_morn and xc_norm are concatenated and used for DL model
                y_norm = torch.from_numpy(
                    self.y[basin, time - warmup: time + rho + horizon, :]
                ).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[basin, :]).float()
            else:
                z_train = xc_norm.float()
            x_train = self.x_origin[basin, time - warmup: time + rho + horizon, :]
            y_train = self.y_origin[basin, time: time + rho + horizon, :]
        else:
            x_norm = self.x[item, :, :]
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                x_norm = self.train_dataset.x[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                xc_norm = torch.from_numpy(x_norm).float()
            else:
                c_norm = self.c[item, :]
                c_norm = (
                    np.repeat(c_norm, x_norm.shape[0], axis=0)
                    .reshape(c_norm.shape[0], -1)
                    .T
                )
                xc_norm = torch.from_numpy(
                    np.concatenate((x_norm, c_norm), axis=1)
                ).float()
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                # when used as input, warmup_length not included for y
                y_norm = torch.from_numpy(self.train_dataset.y[item, :, :]).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[item, :]).float()
            else:
                z_train = xc_norm
            x_train = self.x_origin[item, :, :]
            y_train = self.y_origin[item, warmup:, :]
        return (
            torch.from_numpy(x_train).float(),
            z_train,
        ), torch.from_numpy(y_train).float()

    def __len__(self):
        return self.num_samples if self.train_mode else len(self.t_s_dict["sites_id"])


class FlexibleDataset(BaseDataset):
    """A dataset whose datasources are from multiple sources according to the configuration"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(FlexibleDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        source_cfgs = self.data_cfgs["source_cfgs"]
        return {
            name: data_sources_dict[name](path)
            for name, path in zip(
                source_cfgs["source_names"], source_cfgs["source_paths"]
            )
        }

    def _read_xyc(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        x_datasets, y_datasets, c_datasets = [], [], []
        gage_ids = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]

        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            if var_name in self.data_cfgs["relevant_cols"]:
                x_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["target_cols"]:
                y_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["constant_cols"]:
                c_datasets.append(
                    data_source_.read_attr_xrdataset(gage_ids, [var_name])
                )

        # 合并所有x, y, c类型的数据集
        x = xr.merge(x_datasets) if x_datasets else xr.Dataset()
        y = xr.merge(y_datasets) if y_datasets else xr.Dataset()
        c = xr.merge(c_datasets) if c_datasets else xr.Dataset()
        if "streamflow" in y:
            area = data_source_.camels.read_area(self.t_s_dict["sites_id"])
            y.update(streamflow_unit_conv(y[["streamflow"]], area))
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            x, y, c
        )

    def _normalize(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            break
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=data_source_.camels,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c


class HydroMeanDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HydroMeanDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )
        return SelfMadeHydroDataset(
            self.data_cfgs["source_cfgs"]["source_path"],
            time_unit=[time_unit],
        )

    def _normalize(self):
        x, y, c = super()._normalize()
        return x.compute(), y.compute(), c.compute()

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["source_cfgs"]["source_path"]["attributes"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __len__(self):
        return self.num_samples

    def _prepare_forcing(self):
        return self._read_from_minio(self.data_cfgs["relevant_cols"])

    def _prepare_target(self):
        return self._read_from_minio(self.data_cfgs["target_cols"])

    def _read_from_minio(self, var_lst):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        interval = self.data_cfgs["min_time_interval"]
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )

        subset_list = []
        for start_date, end_date in t_range:
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H") + timedelta(hours=interval)
            ).strftime("%Y-%m-%d-%H")
            subset = self.data_source.read_ts_xrdataset(
                gage_id_lst,
                t_range=[start_date, adjusted_end_date],
                var_lst=var_lst,
                time_units=[time_unit],
            )
            subset_list.append(subset[time_unit])
        return xr.concat(subset_list, dim="time")


class Seq2SeqDataset(HydroMeanDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(data_cfgs, is_tra_val_te)

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
            x_origin = xr.where(x_origin < 0, float("nan"), x_origin)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_attr_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        prec = self.data_cfgs["prec_window"]

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0].reshape(-1, 1)
        s = self.x[basin, idx: idx + rho, 1:]
        x = np.concatenate((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:], c[rho:]), axis=1)
        y = self.y[basin, idx + rho - prec + 1: idx + rho + horizon + 1, :]

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(x).float(),
                torch.from_numpy(x_h).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class TransformerDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(TransformerDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0]
        s = self.x[basin, idx: idx + rho, 1]
        x = np.stack((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:].reshape(-1, 1), c[rho:]), axis=1)
        y = self.y[basin, idx + rho + 1: idx + rho + horizon + 1, :]

        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class GNNDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(GNNDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.total_graph = None
        self.x_up = None
        self.network_features = gpd.read_file(self.data_cfgs['network_shp'])
        self.node_features = gpd.read_file(self.data_cfgs['node_shp'])
        self.load_data()

    def __getitem__(self, item: int):
        # 从lookup_table中获取的idx和basin是整数，但是total_df的basin是字符串，所以需要转换一下
        # 同时需要注意范围，basin在不在103之内，idx又在哪里
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        # 在这里p和s的间隔应该是1吗？
        p_up = self.x_up[basin, idx + 1: idx + rho + horizon + 1, 0]
        s_up = self.x_up[basin, idx: idx + rho, 1]
        x_ps_up = np.stack((p_up[:rho], s_up), axis=1)
        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((self.x, x_ps_up, c[:rho]), axis=1)
        y = self.y[basin, idx + rho + 1: idx + rho + horizon + 1, :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def load_data(self):
        upstream_df, total_graph = self.get_upstream_df()
        self.x_up = xr.Dataset.from_dataframe(upstream_df)
        self.total_graph = total_graph

    def get_upstream_graph(self):
        # 训练时所有流域内所有站点在给定时间段内的水位径流数据
        import igraph as ig
        import networkx as nx
        total_graph = ig.Graph(directed=True)
        total_node_list = []
        basin_station_dict = {}
        for basin in self.data_cfgs["object_ids"]:
            basin_num = basin.split('_')[1]
            upstream_graph = self.prepare_graph(self.network_features, self.node_features, basin_num)
            if len(upstream_graph) != 0:
                if upstream_graph.dtype == 'O':
                    # upstream_graph = array(list1, list2, list3)
                    if upstream_graph.shape[0] > 1:
                        nodes_arr = np.unique(list(chain.from_iterable(upstream_graph)))
                    else:
                        # upstream_graph = array(list1)
                        nodes_arr = upstream_graph
                else:
                    # upstream_graph = array(list1, list2) and dtype is not object
                    nodes_arr = np.unique(upstream_graph)
                # total_node_list is 1-dim list
                total_node_list.extend(nodes_arr.tolist())
                for node in nodes_arr:
                    basin_station_dict[node] = basin
                nx_graph = nx.DiGraph()
                for path in upstream_graph:
                    if isinstance(path, int | np.int64):
                        nx.add_path(nx_graph, [path])
                    else:
                        nx.add_path(nx_graph, path)
                ig_upstream_graph = ig.Graph.from_networkx(nx_graph)
                total_graph = ig.Graph.disjoint_union(total_graph, ig_upstream_graph)
            else:
                continue
        # ig_graph展成list之后，节点序号也是从小到大排列，与dgl_graph.nodes()排号形成对应，不需要再排序添加数据
        return total_graph, total_node_list, basin_station_dict

    def get_upstream_df(self):
        total_graph, total_node_list, basin_station_dict = self.get_upstream_graph()
        nodes_arr = np.unique(total_node_list)
        total_df = pd.DataFrame()
        for up_node in nodes_arr:
            if 'STCD' in self.node_features.columns:
                up_node_name = self.node_features['STCD'][self.node_features.index == up_node].to_list()[0]
            else:
                up_node_name = self.node_features['ID'][self.node_features.index == up_node].to_list()[0]
            station_df = pd.DataFrame()
            for date_tuple in self.data_cfgs[f"t_range_{self.is_tra_val_te}"]:
                data_table = self.read_data_with_stcd_from_minio(up_node_name)
                date_times = pd.date_range(date_tuple[0], date_tuple[1], freq='1H')
                level_str_df = self.gen_level_stream(data_table, date_times=date_times)
                station_df = pd.concat([station_df, level_str_df])
            basin_column = pd.DataFrame({'basin_id': np.repeat(basin_station_dict[up_node], len(station_df))})
            station_df = pd.concat([basin_column, station_df], axis=1)
            total_df = pd.concat([total_df, station_df], axis=0)
        total_df = total_df.set_index('basin_id')
        return total_df, total_graph

    def prepare_graph(self, network_features: gpd.GeoDataFrame, node_features: gpd.GeoDataFrame, node: int | str,
                      cutoff=2147483647):
        # test_df_path = 's3://stations-origin/zq_stations/hour_data/1h/zq_CHN_songliao_10800300.csv'
        if isinstance(node, int):
            node_idx = node
        else:
            if 'STCD' in node_features.columns:
                node_features['STCD'] = node_features['STCD'].astype(str)
                node_idx = node_features.index[node_features['STCD'] == node]
            else:
                node_features['ID'] = node_features['ID'].astype(str)
                node_idx = node_features.index[node_features['ID'] == node]
        if len(node_idx) != 0:
            node_idx = node_idx.tolist()[0]
            try:
                graph_lists = htip.find_edge_nodes(node_features, network_features, node_idx, 'up', cutoff)
            except IndexError:
                # 部分点所对应的LineString为空，导致报错
                graph_lists = []
        else:
            graph_lists = []
        return graph_lists

    def read_data_with_stcd_from_minio(self, stcd: str):
        import hydrodatasource.configs.config as hdscc
        minio_path_zq_chn = f's3://stations-origin/zq_stations/hour_data/1h/zq_CHN_songliao_{stcd}.csv'
        minio_path_zz_chn = f's3://stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_{stcd}.csv'
        minio_path_zq_usa = f's3://stations-origin/zq_stations/hour_data/1h/zq_USA_usgs_{stcd}.csv'
        minio_path_zz_usa = f's3://stations-origin/zz_stations/hour_data/1h/zz_USA_usgs_{stcd}.csv'
        minio_path_zq_usa_new = f's3://stations-origin/zq_stations/hour_data/1h/usgs_datas_462_basins_after_2019/zz_USA_usgs_{stcd}.csv'
        camels_hourly_files = f's3://datasets-origin/camels-hourly/data/usgs_streamflow_csv/{stcd}-usgs-hourly.csv'
        minio_data_paths = [minio_path_zq_chn, minio_path_zz_chn, minio_path_zq_usa, minio_path_zz_usa,
                            minio_path_zq_usa_new, camels_hourly_files]
        hydro_df = None
        for data_path in minio_data_paths:
            if hdscc.FS.exists(data_path):
                hydro_df = pd.read_csv(data_path, engine='c', storage_options=hdscc.MINIO_PARAM)
                break
        if hydro_df is None:
            interim_df = pd.read_sql(f"SELECT * FROM ST_RIVER_R WHERE stcd = '{stcd}'", hdscc.PS)
            if len(interim_df) == 0:
                interim_df = pd.read_sql(f"SELECT * FROM ST_RSVR_R WHERE stcd = '{stcd}'", hdscc.PS)
            hydro_df = interim_df
        return hydro_df

    def gen_level_stream(self, data_table: pd.DataFrame, date_times: pd.DatetimeIndex):
        table, freq = self.df_resample_cut(data_table)
        if len(table) != 0:
            if 'rz' in table.columns:
                table = table.rename(columns={'rz': 'Z', 'inq': 'Q'})
            # 水位(00065)，流量(00060), https://waterservices.usgs.gov/docs/site-service/site-service-details/
            lev_col_name = 'Z' if 'Z' in table.columns else ('z' if 'z' in table.columns else '00065')
            level_array = table[lev_col_name][table.index.isin(date_times)].to_numpy()
            stream_col_name = 'streamflow' if 'streamflow' in table.columns else \
                ('Q' if 'Q' in table.columns else ('q' if 'q' in table.columns else '00060'))
            streamflow_array = table[stream_col_name][table.index.isin(date_times)].to_numpy()
            level_stream_df = pd.DataFrame({'streamflow': streamflow_array, 'level': level_array})
        else:
            level_stream_df = pd.DataFrame()
        return level_stream_df

    def df_resample_cut(self, cut_df: pd.DataFrame):
        time_str = self.step_mode(cut_df)
        tm_col = 'TM' if 'TM' in cut_df.columns else 'tm'
        if '0 days' in time_str:
            time_only_str = time_str.split(' ')[2]
            time_obj = parse(time_only_str).time()
            time_delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second)
            total_minutes = round(time_delta.total_seconds() / 60)
            freq = f'{math.ceil(total_minutes / 60)}h'
            cut_df[tm_col] = pd.to_datetime(cut_df[tm_col])
            try:
                cut_df = cut_df.set_index(tm_col).resample(freq).interpolate().dropna(how='all')
            except ValueError:
                # such as STCD 11210118, quality of data is very bad, pass it
                cut_df = pd.DataFrame()
        else:
            freq = 'D'
            cut_df[tm_col] = pd.to_datetime(cut_df[tm_col])
            try:
                cut_df = cut_df.set_index(tm_col).resample(freq, origin='start').interpolate().dropna(how='all')
            except ValueError:
                cut_df = pd.DataFrame()
        return cut_df, freq

    def step_mode(self, csv_df):
        tm_col = 'TM' if 'TM' in csv_df.columns else 'tm'
        diffs = pd.to_datetime(csv_df[tm_col]).diff()
        # 数据过少，没有众数，当作1天间隔
        if diffs.shape[0] <= 1:
            return '1 days 00:00:00'
        else:
            time_diff = diffs.mode()[0]
            return str(time_diff)
