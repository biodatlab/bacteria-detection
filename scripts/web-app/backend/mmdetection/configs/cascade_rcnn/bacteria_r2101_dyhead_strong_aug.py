_base_ = ['./bacteria_cascade_r50.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1, 
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    
]

img_scale = (1333, 800)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
image_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing'
train_coco_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/split_coco_jbing/Merge_train_jbing.json'
test_coco_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/split_coco_jbing/Merge_test_jbing.json'
val_coco_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/split_coco_jbing/Merge_val_jbing.json'
CLASSES = ("unt", "Amp", "Cip", "Rif", "Tet", "Col")


train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
        dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc', # use format = 'pascal_voc for evey dataset format <3
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

rpn_weight = 0.7
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    
    neck=[dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)]
    
    
    )
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,)
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step= [50, 80])

# runtime settings
evaluation = dict(save_best='auto',
    interval=1, 
    metric='bbox',
    # ap each class
    classwise=True)

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)

# load weight
load_from = "./work_dirs/crcnn_r2101_dyhead_7-2-23/best_bbox_mAP_epoch_110.pth"

# resume from last epoch
# resume_from = "./work_dirs/crcnn_r2101_dyhead_strong_aug_no_bc_14-2-23/epoch_40.pth"

# save dir
work_dir = "./work_dirs/crcnn_r2101_dyhead_strong_aug_no_bc_14-2-23"