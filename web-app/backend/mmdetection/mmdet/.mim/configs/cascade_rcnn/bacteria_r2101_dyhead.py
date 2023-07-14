_base_ = ['./bacteria_albu.py']

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

evaluation = dict(save_best='auto',
    interval=1, 
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=5)
