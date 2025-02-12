import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.disp_predictors import build_disp_predictor
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator

eps = 1e-8

class HODC_PSMNet(nn.Module):
    """
    Hierarchical Object-Aware Dual-Level Contrastive Learning (HODC) for PSMNet

    """
    def __init__(self, cfg):
        super(HODC_PSMNet, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.backbone = build_backbone(cfg)
        self.cost_processor = build_cost_processor(cfg)
        self.disp_predictor = build_disp_predictor(cfg)
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

        self.loss_stereo_ins = DualLevelContrastiveLoss(dim=32)

    def forward(self, batch, epoch=-1, max_epochs=-1, cov_list=None):
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target_l = batch['leftDisp'] if 'leftDisp' in batch else None
        target_r = batch['rightDisp'] if 'rightDisp' in batch else None
        left_object_index = batch['leftObjectIndex'] if 'leftObjectIndex' in batch else None
        right_object_index = batch['rightObjectIndex'] if 'rightObjectIndex' in batch else None
        
        # extract image feature
        left_fms, right_fms = self.backbone(ref_img, tgt_img)

        if isinstance(left_fms, list):
            ref_fms = left_fms[0]
            tgt_fms = right_fms[0]
        else:
            ref_fms, tgt_fms = left_fms, right_fms
        

        # compute cost volume
        costs = self.cost_processor(ref_fms, tgt_fms)

        # disparity prediction
        disps = [self.disp_predictor(cost) for cost in costs]

        if self.training:
            loss_dict = dict()
            
            gsm_loss_dict = self.loss_evaluator(disps, costs, target_l)
            loss_dict.update(gsm_loss_dict)
                
            scale_fms = target_l.shape[-1] / ref_fms.shape[-1]
            scale_ori = random.choice([0.25, 0.5, 1.0])
            
            ref_fms = F.interpolate(ref_fms, scale_factor=scale_fms * scale_ori, mode='bilinear')
            tgt_fms = F.interpolate(tgt_fms, scale_factor=scale_fms * scale_ori, mode='bilinear')
            disps_interp_left = F.interpolate(target_l, scale_factor=scale_ori, mode='bilinear') * scale_ori
            disps_interp_right = F.interpolate(target_r, scale_factor=scale_ori, mode='bilinear') * scale_ori
            index_interp_left = F.interpolate(left_object_index, scale_factor=scale_ori)
            index_interp_right = F.interpolate(right_object_index, scale_factor=scale_ori)

            # contrast_loss_instance = self.loss_stereo_ins(ref_fms, tgt_fms, index_interp_left, index_interp_right, disps_interp_left, disps_interp_right, weight=(5.0 - 2.5 * epoch / max_epochs))
            contrast_loss_instance = self.loss_stereo_ins(ref_fms, tgt_fms, index_interp_left, index_interp_right, disps_interp_left, disps_interp_right, weight=5.0)
            loss_dict.update(contrast_loss_instance)
            
            return {}, loss_dict

        else:
            results = dict(
                ref_fms=[ref_fms],
                tgt_fms=[tgt_fms],
                disps=disps,
                costs=costs,
            )

            return results, {}    

class DualLevelContrastiveLoss(nn.Module):
    def __init__(self, dim):
        super(DualLevelContrastiveLoss, self).__init__()
        self.dim = dim

    def forward(self, ref_fms, tgt_fms, left_object_index, right_object_index, left_disp, right_disp, proportion=1.0, weight=1.0):
        b, c, h, w = ref_fms.size()
        b_1, c_1, h_1, w_1 = left_object_index.size()
        b_2, c_2, h_2, w_2 = left_disp.size()
        assert ref_fms.size() == tgt_fms.size()
        assert left_object_index.size() == right_object_index.size()
        assert left_disp.size() == right_disp.size()
        assert b == b_1 and h == h_1 and w == w_1
        assert b == b_2 and h == h_2 and w == w_2
        assert c_1 == 1 and c_2 == 1

        index_l = left_object_index.clone().repeat(1, c, 1, 1).long()
        index_r = right_object_index.clone().repeat(1, c, 1, 1).long()

        loss = dict()
        
        feat_l2r, feat_r2l = warp(ref_fms, right_disp), warp(tgt_fms, -left_disp)
        mask_occ_l2r = get_occ_mask(-right_disp, -left_disp).repeat(1, c, 1, 1).bool()
        mask_occ_r2l = get_occ_mask(left_disp, right_disp).repeat(1, c, 1, 1).bool()

        reduce_method = 'mean'
        
        base_grid_size = []
        for x in [2, 4, 8, 16]:
            for y in [2, 4, 8, 16, 32]:
                if x * y <= 128:
                    base_grid_size.append((x, y))
        
        num_h, num_w = random.choice(base_grid_size)
        factor = random.choice([2, 4])
        loss_base, keys_base = get_grid_loss(feat_l2r, tgt_fms, mask_occ_l2r, index_r, num_h, num_w, 0.05, weight=weight, proportion=proportion, grid_reduce=reduce_method)
        loss_sub, keys_sub = get_grid_loss(feat_l2r, tgt_fms, mask_occ_l2r, index_r, num_h * factor, num_w * factor, 0.05, weight=weight, proportion=proportion, global_rep=keys_base, grid_reduce=reduce_method)
        
        loss['loss_dual_level_contrastive'] = (loss_base + loss_sub) / 2.
        
        del keys_base
        del keys_sub
        del feat_l2r
        del feat_r2l
        torch.cuda.empty_cache()
        
        
        return loss
        
def get_grid_loss(feat_query, feat_key, mask_occ, index, num_h, num_w, T, weight=1.0, proportion=1.0, additional_keys=None, global_rep=None, grid_reduce='mean'):
    b, c, h, w = feat_query.size()
    assert feat_query.size() == feat_key.size() and feat_query.size() == mask_occ.size() and feat_query.size() == index.size()
    
    feat_query = feat_query.reshape(b, c, h * w)
    feat_key = feat_key.reshape(b, c, h * w)
    mask_occ = mask_occ.reshape(b, c, h * w)
    index = index.reshape(b, c, h * w)
    
    if global_rep is not None:
        assert feat_query.size() == global_rep.size()
    
    _grid_num = torch.linspace(0, num_h * num_w - 1, num_h * num_w, dtype=torch.long).reshape(-1, 1).repeat(1, w // num_w).reshape(-1, w // num_w * num_w).repeat(1, h // num_h).reshape(1, 1, h // num_h * num_h, w // num_w * num_w).cuda()
    # pad extra area with zero
    grid_num = torch.zeros(1, 1, h, w).cuda()
    grid_num[:, :, :h // num_h * num_h, :w // num_w * num_w] = _grid_num
    grid_num = grid_num.reshape(1, 1, h * w).repeat(b, c, 1).long()
    
        
    index_grid = index * num_h * num_w + grid_num
    cat_num_grid = torch.max(index_grid) + 1 + 1
    reduced_query_grid = torch.zeros((b, c, cat_num_grid), dtype=feat_query.dtype).cuda()
    reduced_key_grid = torch.zeros((b, c, cat_num_grid), dtype=feat_key.dtype).cuda()
        
    # let the occluded area belongs to the same category, i.e., maximum + 1, and discard later
    index_grid[~mask_occ] = cat_num_grid - 1

        
    reduced_query_grid = reduced_query_grid.scatter_reduce(dim=2, index=index_grid, src=feat_query, reduce=grid_reduce)
    reduced_key_grid = reduced_key_grid.scatter_reduce(dim=2, index=index_grid, src=feat_key, reduce=grid_reduce)
    
    gathered_keys = torch.gather(input=reduced_key_grid, dim=2, index=index_grid)
    
    reduced_query_grid = reduced_query_grid.permute(0, 2, 1)[:, :-1, :].reshape(b * (cat_num_grid - 1), c)
    reduced_key_grid = reduced_key_grid.permute(0, 2, 1)[:, :-1, :].reshape(b * (cat_num_grid - 1), c)
        
    mask_valid_grid = torch.logical_and(reduced_query_grid.sum(dim=1) != 0, reduced_key_grid.sum(dim=1) != 0)
    valid_cat_num_grid = mask_valid_grid.sum().long()
    
    reduced_query_grid, reduced_key_grid = reduced_query_grid[mask_valid_grid], reduced_key_grid[mask_valid_grid]
    assert valid_cat_num_grid == reduced_query_grid.size(0) and valid_cat_num_grid == reduced_key_grid.size(0)
        
    query = F.normalize(reduced_query_grid, dim=1)
    if additional_keys is not None:
        keys = torch.cat([F.normalize(reduced_key_grid, dim=1), F.normalize(additional_keys.T, dim=1)], dim=0)
    else:
        keys = F.normalize(reduced_key_grid, dim=1)
        
    logits = torch.mm(query, keys.permute(1, 0)) / T
    pos = logits.diag()
    # at least two items on the dominator
    neg = torch.sort(logits).values[:, min(int(logits.size(1) * (1.0 - proportion)), logits.size(1) - 3):]
    loss = -torch.log(torch.exp(pos) / torch.exp(neg).sum(dim=1)).mean() * weight
    
    if global_rep is not None:
        reduced_global_key = torch.zeros((b, c, cat_num_grid), dtype=global_rep.dtype).cuda()
        reduced_global_key = reduced_global_key.scatter_reduce(dim=2, index=index_grid, src=global_rep, reduce=grid_reduce)
        reduced_global_key = reduced_global_key.permute(0, 2, 1)[:, :-1, :].reshape(b * (cat_num_grid - 1), c)
        reduced_global_key = reduced_global_key[mask_valid_grid]
        assert valid_cat_num_grid == reduced_global_key.size(0)
        
        logits_local_global = torch.mm(query, F.normalize(reduced_global_key, dim=1).T) / T
        pos = logits_local_global.diag()
        neg = torch.sort(logits_local_global).values[:, min(int(logits.size(1) * (1.0 - proportion)), logits.size(1) - 3):]
        loss += -torch.log(torch.exp(pos) / torch.exp(neg).sum(dim=1)).mean() * weight

    return loss, gathered_keys

def warp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()
    device = disp.device
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img).to(device)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img).to(device)

    # Apply shift in X direction
    x_shifts = (disp[:, 0, :, :] / w).to(device)
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output


def get_occ_mask(left_disp, right_disp, threshold=3.0):
    b, _, h, w = left_disp.size()
    device = left_disp.device
    index = torch.arange(w).float().to(device)
    index = index.repeat(b, 1, h, 1)
    index_l2r = warp(index, right_disp)
    index_l2r2l = warp(index_l2r, -left_disp)

    masko = torch.abs(index - index_l2r2l) < threshold

    return masko.float()