_base_ = './bacteria_r2101dcn_dyhead.py'

model = dict(
    backbone=dict(
        stem_channels=128,
        depth=50,
        init_cfg=dict(type='Pretrained',                   
                      checkpoint='open-mmlab://resnest50')),
    roi_head=dict(bbox_head=[
        dict(
            type='SABLHead',
            num_classes=19,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
        dict(
            type='SABLHead',
            num_classes=19,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.5),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
        dict(
            type='SABLHead',
            num_classes=19,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.3),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0))
    ])
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

image_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing'
# coco_path = './ann_coco/all_bacteria_annotation_coco.json'
train_coco_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/split_coco_jbing/merge_db_train.json'
test_coco_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/split_coco_jbing/merge_db_test.json'
CLASSES = ('unt', 'Amp', 'Cip', 'Rif', 'Tet', 'Col', 'kan', 'Mec', 'Control',
       'Oblique', 'Vesicle', 'Nalidixate', 'MP265', 'Mecillinam', 'Rod',
       'Dividing', 'Rifampicin', 'Microcolony', 'CAM')

data = dict(

    samples_per_gpu=1,
    workers_per_gpu=1,
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
        ann_file=test_coco_path, # change to val to re-collect the data
        img_prefix=image_path ,
        classes=CLASSES,
        pipeline=test_pipeline)
    )
# 
# load_from = "./work_dirs/crcnn_r2101_dcnv2_8class_24-2-23/best_bbox_mAP_epoch_52.pth"
work_dir = './work_dirs/crcnn_r250_dcnv2_dyhead_pretrain_kaggle_4-3-23'
load_from = "./work_dirs/crcnn_r250_dcnv2_dyhead_8class_26-2-23/best_bbox_mAP_epoch_58.pth"