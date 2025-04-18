import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
import os
import os.path as osp

from mmcv import mkdir_or_exist
from dmb.visualization.stereo.show_result import ShowResultTool


class SaveResultTool(object):
    def __call__(self, result, out_dir, image_name):
        result_tool = ShowResultTool()
        result = result_tool(result, color_map='gray', bins=100)

        # if 'GrayDisparity' in result.keys():
        #     grayEstDisp = result['GrayDisparity']
        #     gray_save_path = osp.join(out_dir, 'disp_0')
        #     mkdir_or_exist(gray_save_path)
        #     skimage.io.imsave(osp.join(gray_save_path, image_name), (grayEstDisp * 256).astype('uint16'))

        if 'ColorDisparity' in result.keys():
            colorEstDisp = result['ColorDisparity']
            color_save_path = osp.join(out_dir, 'color_disp')
            mkdir_or_exist(color_save_path)
            plt.imsave(osp.join(color_save_path, image_name), colorEstDisp, cmap=plt.cm.hot)

        if 'GroupColor' in result.keys():
            group_save_path = os.path.join(out_dir, 'group_disp')
            mkdir_or_exist(group_save_path)
            plt.imsave(osp.join(group_save_path, image_name), result['GroupColor'], cmap=plt.cm.hot)

        if 'ColorConfidence' in result.keys():
            conf_save_path = os.path.join(out_dir, 'confidence')
            mkdir_or_exist(conf_save_path)
            plt.imsave(osp.join(conf_save_path, image_name), result['ColorConfidence'])
