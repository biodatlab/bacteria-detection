_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/cascade_rcnn/bacteria_albu.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))


runner = dict(type='EpochBasedRunner', max_epochs=400)
evaluation = dict(save_best='auto',
    interval=10, 
    metric='bbox')
checkpoint_config = dict(interval=40)