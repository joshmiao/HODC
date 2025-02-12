import random
import numbers
from typing import Any
import numpy as np

import torch
from torch.nn.functional import pad
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter




class ToTensor(object):
    """
    convert numpy.ndarray to torch.floatTensor, in [Channels, Height, Width]
    """
    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy())
        return sample


class ToArray(object):
    """
    """
    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], torch.Tensor):
                sample[k] = np.array(sample[k])
        return sample


class CenterCrop(object):
    """Crops the given image at central location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class RandomCrop(object):
    """Crops the given image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['leftImage'] = F.normalize(
            sample['leftImage'], mean=self.mean, std=self.std
        )
        sample['rightImage'] = F.normalize(
            sample['rightImage'], mean=self.mean, std=self.std
        )
        # sample['rightImage_c'] = F.normalize(
        #     sample['rightImage_c'], mean=self.mean, std=self.std
        # )
        # sample['raw_leftImage'] = F.normalize(
        #     sample['raw_leftImage'], mean=self.mean, std=self.std
        # )
        # sample['raw_rightImage'] = F.normalize(
        #     sample['raw_rightImage'], mean=self.mean, std=self.std
        # )
        return sample


class StereoPad(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        base = 64

        if h > th:
            th = ((h // base) + 1) * base
        if w > tw:
            tw = ((w // base) + 1) * base

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0

        sample['leftImage'] = pad(
            sample['leftImage'], [pad_left, pad_right, pad_top, pad_bottom],
            mode='constant', value=0
        )
        sample['rightImage'] = pad(
            sample['rightImage'], [pad_left, pad_right, pad_top, pad_bottom],
            mode='constant', value=0
        )
        # sample['rightImage_c'] = pad(
        #     sample['rightImage_c'], [pad_left, pad_right, pad_top, pad_bottom],
        #     mode='constant', value=0
        # )

        return sample



class RAW(object):
    def __init__(self):
        self.raw = True

    def __call__(self, sample):
        sample['raw_leftImage'] = sample['leftImage'].copy()
        sample['raw_rightImage'] = sample['rightImage'].copy()
        # sample['rightImage_c'] = sample['rightImage'].copy()

        return sample

class StereoAugmentation(object):
    def __init__(self, data):
        self.data = data
        self.color_jitter = None
        self.adjust_gamma = None
        self.pixel_noise = None
        self.blur = None
        self.erase = None
        if 'augmentations' in self.data:
            if 'color_jitter' in self.data.augmentations:
                color_jitter_cfg = self.data.augmentations.color_jitter
                brightness = color_jitter_cfg.get('brightness', 0)
                contrast = color_jitter_cfg.get('contrast', 0)
                saturation = color_jitter_cfg.get('saturation', 0)
                hue = color_jitter_cfg.get('hue', 0) 
                prob = color_jitter_cfg.get('prob', 0)
                gamma = color_jitter_cfg.get('gamma', 0)
                self.color_jitter = ImageColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, prob=prob)
                self.adjust_gamma = ImageAdjustGamma(gamma_min=1.0 - gamma, gamma_max=1.0 + gamma, prob=prob)
                
            if 'pixel_noise' in self.data.augmentations:
                pixel_noise_cfg = self.data.augmentations.pixel_noise
                mean = pixel_noise_cfg.get('mean', 0)
                var = pixel_noise_cfg.get('var', 0)
                prob = pixel_noise_cfg.get('prob', 0)
                self.pixel_noise = ImagePixelNoise(mean=mean, var=var, prob=prob)
                
            if 'blur' in self.data.augmentations:
                blur_cfg = self.data.augmentations.blur
                min_factor = blur_cfg.get('min_factor', 1)
                prob = blur_cfg.get('prob', 0)
                self.blur = ImageBlur(min_factor=min_factor, prob=prob)
                
            if 'erase' in self.data.augmentations:
                erase_cfg = self.data.augmentations.erase
                size = erase_cfg.get('size', 0)
                max_nums = erase_cfg.get('max_nums', 0)
                prob = erase_cfg.get('prob', 0)
                self.erase = RandomEraser(size=size, max_nums=max_nums, prob=prob)
                
            
    def __call__(self, sample):
        if self.color_jitter is not None:
            sample = self.color_jitter(sample)

        if self.adjust_gamma is not None:
            sample = self.adjust_gamma(sample)

        if self.pixel_noise is not None:
            sample = self.pixel_noise(sample)
            
        if self.blur is not None:
            sample = self.blur(sample)
            
        if self.erase is not None:
            sample = self.erase(sample)
            
        return sample

class ImageColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob
        self.argumentor = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            sample['leftImage'] = self.argumentor(sample['leftImage'].to(torch.uint8)).float()
            sample['rightImage'] = self.argumentor(sample['rightImage'].to(torch.uint8)).float()
        return sample
    
class ImagePixelNoise(object):
    def __init__(self, mean, var, prob):
        self.mean = mean
        self.var = var
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            sample['leftImage'] = sample['leftImage'] + (torch.randn_like(sample['leftImage']) * self.var + self.mean).to(torch.uint8)
            sample['leftImage'] = torch.clamp(sample['leftImage'], 0, 255)
            
            sample['rightImage'] = sample['rightImage'] + (torch.randn_like(sample['rightImage']) * self.var + self.mean).to(torch.uint8)
            sample['rightImage'] = torch.clamp(sample['rightImage'], 0, 255)
        return sample


class RandomEraser(object):
    def __init__(self, size, max_nums, prob):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.max_nums = max_nums
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            sample['rightImage'] = self.eraser_transform(sample['rightImage'])
        return sample

    def eraser_transform(self, img):
        h, w = img.shape[-2:]
        assert h > self.size[0] and w > self.size[1], 'cropped image size is smaller than the area that should be erased (cropped image size: [{0:},{1:}], erase size: [{2:}, {3:}])'.format(h, w, self.size[0], self.size[1])
        
        mean_color = torch.mean(img.reshape(3, -1), dim=1)
        for _ in range(np.random.randint(self.max_nums + 1)):
            x0 = np.random.randint(0, w - self.size[1])
            y0 = np.random.randint(0, h - self.size[0])
            dx = np.random.randint(self.size[1] // 2, self.size[1])
            dy = np.random.randint(self.size[0] // 2, self.size[0])
            img[0, y0:y0 + dy, x0:x0 + dx] = mean_color[0]
            img[1, y0:y0 + dy, x0:x0 + dx] = mean_color[1]
            img[2, y0:y0 + dy, x0:x0 + dx] = mean_color[2]
        return img

class ImageBlur(object):
    def __init__(self, min_factor, prob):
        self.min_factor = min_factor
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            factor = random.uniform(self.min_factor, 1.0)
            sample['leftImage'] = self.blur(sample['leftImage'], factor)
            sample['rightImage'] = self.blur(sample['rightImage'], factor)
        return sample

    def blur(self, img, factor):
        h, w = img.shape[-2:]
        img = F.resize(img, [int(h * factor), int(w * factor)], antialias=True)
        img = F.resize(img, [h, w], antialias=True)
        return img

class ImageAdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, prob, gain_min=1.0, gain_max=1.0):
        self.prob = prob
        self.argumentor = AdjustGamma(gamma_min, gamma_max, gain_min, gain_max)

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            sample['leftImage'] = self.argumentor(sample['leftImage'].to(torch.uint8)).float()
            sample['rightImage'] = self.argumentor(sample['rightImage'].to(torch.uint8)).float()
        return sample


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(sample, gamma, gain)