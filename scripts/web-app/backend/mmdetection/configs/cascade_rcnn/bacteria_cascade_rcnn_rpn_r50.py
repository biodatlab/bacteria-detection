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
        num_outs=5)
    )

evaluation = dict(save_best='auto',
    interval=10, 
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=300)
