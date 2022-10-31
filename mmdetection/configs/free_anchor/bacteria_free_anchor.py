_base_ = '../retinanet/bacteria_retinanet_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='FreeAnchorRetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

image_path = '/home/badboy-002/github/senior_project/'
# coco_path = './ann_coco/all_bacteria_annotation_coco.json'
train_coco_path = './ann_coco/Train_bacteria_annotation_coco.json'
test_coco_path = './ann_coco/Test_bacteria_annotation_coco.json'
CLASSES = ("Amp", "Cip", "Rif", "Tet", "unt")

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=train_coco_path,
        img_prefix=image_path ,
        classes=CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=test_coco_path,
        img_prefix=image_path,  
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=test_coco_path,
        img_prefix=image_path ,
        classes=CLASSES,
        pipeline=test_pipeline)
    )

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]) 
runner = dict(type='EpochBasedRunner', max_epochs= 100)
evaluation = dict(interval=20, metric='bbox')
checkpoint_config = dict(interval=50)

