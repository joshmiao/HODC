import os.path as osp

# the task of the model for, including 'stereo' and 'flow', default 'stereo'
task = 'stereo'

# model settings
max_disp = 192
model = dict(
    meta_architecture="HODC",
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

root = '/root/autodl-tmp/HODC/work_dir/'

sceneflow_data_root = osp.join(root, 'StereoMatching', 'SceneFlow')
sceneflow_annfile_root = osp.join(root, 'StereoMatching/annotations', 'SceneFlow_object')

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
    sparse=True,
    imgs_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='SceneFlow',
        data_root=sceneflow_data_root,
        annfile=osp.join(sceneflow_annfile_root, 'finalpass_train.json'),
        input_shape=[256, 512],
        # input_shape=[320, 736],
        augmentations=dict(            
            color_jitter=dict(brightness=[0.5, 2.0], contrast=0.5, saturation=[0, 1.4], hue=1 / 3.14, gamma=0.5, prob=1.0),
            # erase=dict(size=(50, 100), max_nums=10, prob=0.5),
        ),
        use_right_disp=False,
        **img_norm_cfg,
    ),
    eval=[
        dict(
            type='KITTI-2015',
            data_root=kitti2015_data_root,
            annfile=osp.join(kitti2015_annfile_root, 'full_train.json'),
            input_shape=[384, 1248],
            use_right_disp=False,
            **img_norm_cfg,
        ),
        dict(
            type='KITTI-2012',
            data_root=kitti2012_data_root,
            annfile=osp.join(kitti2012_annfile_root, 'full_train.json'),
            input_shape=[384, 1248],
            use_right_disp=False,
            **img_norm_cfg,
        ),
        dict(
            type='Middlebury',
            data_root=midd_data_root,
            annfile=osp.join(midd_annfile_root, 'trainH.json'),
            input_shape=[0, 0],
            use_right_disp=False,
            **img_norm_cfg,
        ),
        dict(
            type='ETH3D',
            data_root=eth3d_data_root,
            annfile=osp.join(eth3d_annfile_root, 'train.json'),
            input_shape=[0, 0],
            use_right_disp=False,
            **img_norm_cfg,
        ),
    ],
    # vis=[
    #     dict(
    #         type='KITTI-2015',
    #         data_root=kitti2015_data_root,
    #         annfile=osp.join(kitti2015_annfile_root, 'full_train.json'),
    #         input_shape=[384, 1248],
    #         use_right_disp=False,
    #         **img_norm_cfg,
    #     ),
    #     dict(
    #         type='KITTI-2012',
    #         data_root=kitti2012_data_root,
    #         annfile=osp.join(kitti2012_annfile_root, 'full_train.json'),
    #         input_shape=[384, 1248],
    #         use_right_disp=False,
    #         **img_norm_cfg,
    #     ),
    #     dict(
    #         type='Middlebury',
    #         data_root=midd_data_root,
    #         annfile=osp.join(midd_annfile_root, 'trainH.json'),
    #         input_shape=[0, 0],
    #         use_right_disp=False,
    #         **img_norm_cfg,
    #     ),
    #     dict(
    #         type='ETH3D',
    #         data_root=eth3d_data_root,
    #         annfile=osp.join(eth3d_annfile_root, 'train.json'),
    #         input_shape=[0, 0],
    #         use_right_disp=False,
    #         **img_norm_cfg,
    #     ),
    # ],
)

optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    gamma=0.5,
    step=[15, 30],
)
checkpoint_config = dict(
    interval=1
)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)


# https://nvidia.github.io/apex/amp.html
apex = dict(
    # whether to use apex.synced_bn
    synced_bn=True,
    # whether to use apex for mixed precision training
    use_mixed_precision=False,
    # the model weight type: float16 or float32
    type="float16",
    # the factor when apex scales the loss value
    loss_scale=16,
)

total_epochs = 45

# each model will return several disparity maps, but not all of them need to be evaluated
# here, by giving indexes, the framework will evaluate the corresponding disparity map
eval_disparity_id = [0, 1, 2]

dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
eval_interval = 1
vis_interval = 10
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'StereoMatching', 'exps/PSMNet/HODC-PSMNet')

# seperate encoder
find_unused_parameters = False
