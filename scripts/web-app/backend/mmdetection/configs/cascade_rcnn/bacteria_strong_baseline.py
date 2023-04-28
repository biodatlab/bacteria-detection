_base_ = ['./bacteria_albu.py']

# custom_imports = dict(
#     imports = ['mmdet.models.backbones.CBNet'],
#     allow_failed_imports=False)
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
    
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    )
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step= [100,200,250])

evaluation = dict(save_best='auto',
    interval=10, 
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=300)
work_dir = './work_dirs/crcnn_r2101_pafpn_pretrain_18-2-23'
