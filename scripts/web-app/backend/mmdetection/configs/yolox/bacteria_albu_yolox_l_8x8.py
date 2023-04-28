_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/yolox/bacteria_albu_yolox.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),

    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))


runner = dict(type='EpochBasedRunner', max_epochs=300)
load_from = "./work_dirs/yolox_m_1-3-23/best_bbox_mAP_epoch_800.pth"
work_dir = './work_dirs/yolox_l_14-3-23'