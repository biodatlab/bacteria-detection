_base_ = ['./bacteria_cascade_r50_pretrain.py']

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
rpn_weight = 0.7
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s'))
    
    )
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step= [100,150,200,250])
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
evaluation = dict(save_best='auto',
    interval=1, 
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
# load_from = "./work_dirs/crcnn_r2101_dyhead_7-2-23/best_bbox_mAP_epoch_110.pth"
# resume_from = "./work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/best_bbox_mAP_epoch_16.pth"
work_dir = './work_dirs/crcnn_r2101_pretrain_19-2-23'
load_from = "./work_dirs/cascade_r50_pretrain_19-2-23/epoch_10.pth"