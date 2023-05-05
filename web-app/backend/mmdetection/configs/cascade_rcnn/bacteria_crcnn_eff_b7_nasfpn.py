_base_ = ['./bacteria_albu.py']

rpn_weight = 0.7
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    
    neck=dict(type='NASFPN', stack_times=7, norm_cfg=dict(type='BN', requires_grad=True),
              in_channels=[48, 136, 384]),
    
    )

data = dict(

    samples_per_gpu=1,
    workers_per_gpu=1,)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step= [100,150,200,250])

evaluation = dict(save_best='auto',
    interval=10, 
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=300)
