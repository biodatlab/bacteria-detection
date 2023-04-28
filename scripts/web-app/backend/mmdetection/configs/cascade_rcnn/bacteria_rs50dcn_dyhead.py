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
            num_classes=8,
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
            num_classes=8,
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
            num_classes=8,
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

# load_from = "./work_dirs/crcnn_r2101_dcnv2_8class_24-2-23/best_bbox_mAP_epoch_52.pth"
work_dir = './work_dirs/kfold/crcnnr250/fold0_18-3-23'
resume_from = None
load_from = None
# load_from = "./work_dirs/crcnn_r250_dcnv2_dyhead_8class_26-2-23/best_bbox_mAP_epoch_58.pth"