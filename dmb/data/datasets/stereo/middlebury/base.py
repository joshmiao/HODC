import os.path as osp
import numpy as np
from imageio import imread
from PIL import Image

from dmb.data.datasets.stereo.base import StereoDatasetBase
from dmb.data.datasets.utils import load_middlebury_disp


class MiddleburyDataset(StereoDatasetBase):

    def __init__(self, annFile, root, transform=None):
        super(MiddleburyDataset, self).__init__(annFile, root, transform)

    def Loader(self, item):
        # only take first three RGB channel no matter in RGB or RGBA format
        leftImage = imread(
            osp.join(self.root, item['left_image_path'])
        ).transpose(2, 0, 1).astype(np.float32)[:3]
        rightImage = imread(
            osp.join(self.root, item['right_image_path'])
        ).transpose(2, 0, 1).astype(np.float32)[:3]

        h, w = leftImage.shape[1], leftImage.shape[2]
        original_size = (h, w)

        data = {
            'leftImage': leftImage,
            'rightImage': rightImage,
            'original_size': original_size,
        }
        
        if 'left_disp_map_path' in item.keys() and item['left_disp_map_path'] is not None:
            leftDisp = load_middlebury_disp(
                osp.join(self.root, item['left_disp_map_path'])
            )
            leftDisp = leftDisp[np.newaxis, ...]
            data['leftDisp'] = leftDisp


        if 'right_disp_map_path' in item.keys() and item['right_disp_map_path'] is not None:
            rightDisp = load_middlebury_disp(
                osp.join(self.root, item['right_disp_map_path'])
            )
            rightDisp = rightDisp[np.newaxis, ...]
            data['rightDisp'] = rightDisp
            
        if 'mask_noc' in item.keys() and item['mask_noc'] is not None:
            occMask = np.array(Image.open(
                osp.join(self.root, item['mask_noc'])
            ), dtype=np.float32)
            occMask = occMask[np.newaxis, ...]
            occMask[occMask != 255] = 1.
            occMask[occMask == 255] = 0.
            data['occMask'] = occMask

        # return {
        #     'leftImage': leftImage,
        #     'rightImage': rightImage,
        #     'leftDisp': leftDisp,
        #     'rightDisp': rightDisp,
        #     'original_size': original_size,
        # }
        
        return data

    @property
    def name(self):
        if self.annFile.split('.')[0][-1] in ['F', 'H', 'Q']:
            return 'Middlebury' + self.annFile.split('.')[0][-1]
        return 'Middlebury'
