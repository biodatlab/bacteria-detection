_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/cascade_rcnn/bacteria_albu.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
runner = dict(type='EpochBasedRunner', max_epochs=400)
evaluation = dict(save_best='auto',
    interval=10, 
    metric='bbox')
checkpoint_config = dict(interval=40)