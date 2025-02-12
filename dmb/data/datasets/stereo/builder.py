from dmb.data.transforms import Compose
from dmb.data.transforms import stereo_trans as T

from dmb.data.datasets.stereo.scene_flow import SceneFlowDataset
from dmb.data.datasets.stereo.kitti import Kitti2012Dataset, Kitti2015Dataset
from dmb.data.datasets.stereo.middlebury import MiddleburyDataset
from dmb.data.datasets.stereo.eth3d import ETH3DDataset




def build_transforms(data, is_train):
    input_shape = data.input_shape
    mean = data.mean
    std = data.std
        

    if is_train:
        transform = Compose(
            [
                T.RandomCrop(input_shape),
                T.RAW(),
                T.ToTensor(),
                T.StereoAugmentation(data),
                T.Normalize(mean, std),
            ]
        )
    else:
        transform = Compose(
            [
                T.RAW(),
                T.ToTensor(),
                T.StereoPad(input_shape),
                T.Normalize(mean, std),
            ]
        )

    return transform


def build_single_stereo_dataset(data, Type):
    is_train = True if Type == 'train' else False
    data_root = data.data_root
    data_type = data.type
    annFile = data.annfile
    
    transforms = build_transforms(data, is_train=is_train)

    if 'SceneFlow' in data_type:
        dataset = SceneFlowDataset(annFile, data_root, transforms)
    elif 'KITTI' in data_type:
        if '2012' in data_type:
            dataset = Kitti2012Dataset(annFile, data_root, transforms)
        elif '2015' in data_type:
            dataset = Kitti2015Dataset(annFile, data_root, transforms)
        else:
            raise ValueError("invalid data type: {}".format(data_type))
    elif 'Middlebury' in data_type:
        dataset = MiddleburyDataset(annFile, data_root, transforms)
    elif 'ETH3D' in data_type:
        dataset = ETH3DDataset(annFile, data_root, transforms)
    else:
        raise ValueError("invalid data type: {}".format(data_type))

    return dataset

def build_stereo_dataset(cfg, Type):
    if Type not in cfg.data:
        return None
    
    if isinstance(cfg.data[Type], list):
        dataset = []
        for data in cfg.data[Type]:
            dataset.append(build_single_stereo_dataset(data, Type))
        return dataset
    
    else:        
        return build_single_stereo_dataset(cfg.data[Type], Type)