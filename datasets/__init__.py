from .dtu import MVSDatasetDTU
from .dtu_regnerf import MVSDatasetDTURegNeRF
from .realestate10k import RealEstate10k, RealEstate10kTest
from .realestate10k_subset import RealEstate10kSubset, RealEstate10kSubsetTest
from .ibrnet_mix.google_scanned_objects import GoogleScannedDataset
from .ibrnet_mix.llff import LLFFDataset
from .ibrnet_mix.llff_test import LLFFTestDataset
from .ibrnet_mix.ibrnet_collected import IBRNetCollectedDataset
from .ibrnet_mix.realestate import RealEstateDataset
from .ibrnet_mix.spaces_dataset import SpacesFreeDataset
from .mipnerf360 import MipNeRF360Dataset

datas_dict = {
    'dtu': MVSDatasetDTU,
    'google_scanned': GoogleScannedDataset,
    'dtu_regnerf': MVSDatasetDTURegNeRF,
    'realestate': RealEstate10k,
    'realestate_test': RealEstate10kTest,
    'realestate_subset': RealEstate10kSubset,
    'realestate_subset_test': RealEstate10kSubsetTest,
    'ibrnet_llff': LLFFDataset,
    'ibrnet_llff_test': LLFFTestDataset,
    'ibrnet_collected': IBRNetCollectedDataset,
    'ibrnet_realestate': RealEstateDataset,
    'spaces': SpacesFreeDataset,
    'mipnerf360': MipNeRF360Dataset,
}
