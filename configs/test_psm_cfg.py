import os.path as osp

# the task of the model for, including 'stereo' and 'flow', default 'stereo'
task = 'stereo'

# model settings
max_disp = 192
model = dict(
    meta_architecture="HODC",
    # whitening
    whitening=False,
    # max disparity
    max_disp=max_disp,
    # the model whether or not to use BatchNorm
    batch_norm=True,
    backbone=dict(
        type="FCPSMNet_woMo",
        # the in planes of feature extraction backbone
        in_planes=3,
        m=0.9999,
    ),
    cost_processor=dict(
        # Use the concatenation of left and right feature to form cost volume, then aggregation
        type='Concatenation',
        cost_computation = dict(
            # default cat_fms
            type="default",
            # the maximum disparity of disparity search range under the resolution of feature
            max_disp = int(max_disp // 4),
            # the start disparity of disparity search range
            start_disp = 0,
            # the step between near disparity sample
            dilation = 1,
        ),
        cost_aggregator=dict(
            type="PSMNet",
            # the maximum disparity of disparity search range
            max_disp = max_disp,
            # the in planes of cost aggregation sub network
            in_planes=64,
        ),
    ),
    disp_predictor=dict(
        # default FasterSoftArgmin
        type='FASTER',
        # the maximum disparity of disparity search range
        max_disp = max_disp,
        # the start disparity of disparity search range
        start_disp = 0,
        # the step between near disparity sample
        dilation = 1,
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,

    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
        contrastive_loss=dict(
            type="Default",
        ),
    ),
    eval=dict(
        # evaluate the disparity map within (lower_bound, upper_bound)
        lower_bound=0,
        upper_bound=max_disp,
        # evaluate the disparity map in occlusion area and not occlusion
        eval_occlusion=True,
        # return the cost volume after regularization for visualization
        is_cost_return=False,
        # whether move the cost volume from cuda to cpu
        is_cost_to_cpu=True,
    ),
)

# dataset settings
root = '/root/autodl-tmp/HODC/work_dir/'

kitti2015_data_root = osp.join(root, 'StereoMatching', 'KITTI-2015')
kitti2015_annfile_root = osp.join(root, 'StereoMatching/annotations', 'KITTI-2015')

kitti2012_data_root = osp.join(root, 'StereoMatching', 'KITTI-2012')
kitti2012_annfile_root = osp.join(root, 'StereoMatching/annotations', 'KITTI-2012')

midd_data_root = osp.join(root, 'StereoMatching', 'Middlebury/MiddEval3')
midd_annfile_root = osp.join(root, 'StereoMatching/annotations', 'Middlebury-eval3')

eth3d_data_root = osp.join(root, 'StereoMatching', 'ETH3D')
eth3d_annfile_root = osp.join(root, 'StereoMatching/annotations', 'ETH3D')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
data = dict(
    # whether disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    # sparse=False,
    sparse=False,
    imgs_per_gpu=1,
    workers_per_gpu=6,
    test=dict(
        type='KITTI-2012',
        data_root=kitti2012_data_root,
        annfile=osp.join(kitti2012_annfile_root, 'full_train.json'),
        input_shape=[384, 1248],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    # test=dict(
    #     type='KITTI-2015',
    #     data_root=kitti2015_data_root,
    #     annfile=osp.join(kitti2015_annfile_root, 'full_train.json'),
    #     input_shape=[384, 1248],
    #     use_right_disp=False,
    #     **img_norm_cfg,
    # ),
    # test=dict(
    #     type='Middlebury',
    #     data_root=midd_data_root,
    #     annfile=osp.join(midd_annfile_root, 'trainH.json'),
    #     input_shape=[0, 0],
    #     use_right_disp=False,
    #     **img_norm_cfg,
    # ),
    # test=dict(
    #     type='ETH3D',
    #     data_root=eth3d_data_root,
    #     annfile=osp.join(eth3d_annfile_root, 'train.json'),
    #     input_shape=[0, 0],
    #     use_right_disp=False,
    #     **img_norm_cfg,
    # ),
)

gpus = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True

# For test
checkpoint = osp.join(root, 'StereoMatching', 'exps/checkpoints/HODC-PSMNet.pth')
work_dir = osp.join(root, 'StereoMatching', 'exps/PSMNet/HODC-PSMNet')
out_dir = osp.join(root, 'StereoMatching', 'exps/PSMNet/HODC-PSMNet')