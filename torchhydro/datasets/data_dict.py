"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-02-12 19:09:33
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: \torchhydro\torchhydro\datasets\data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.datasets.data_sets import (
    BaseDataset,
    BasinSingleFlowDataset,
    DplDataset,
    FlexibleDataset,
    HydroGridDataset,
    HydroMeanDataset,
    HydroMultiSourceDataset,
    PrecipitationFusionDataset,
    MopexPrecipitationGagesAttrFusionDataset,
)


datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "GridDataset": HydroGridDataset,
    "MeanDataset": HydroMeanDataset,
    "FlexDataset": FlexibleDataset,
    "MultiSourceDataset": HydroMultiSourceDataset,
    "PrecipitationFusionDataset": PrecipitationFusionDataset,
    "MopexPrecipitationGagesAttrFusionDataset": MopexPrecipitationGagesAttrFusionDataset,
}
