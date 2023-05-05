_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/yolox/bacteria_albu_yolox.py'


# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))


runner = dict(type='EpochBasedRunner', max_epochs=7
              00)
load_from = "./work_dirs/yolox_m_1-3-23/best_bbox_mAP_epoch_800.pth"
work_dir = './work_dirs/yolox_x_14-3-23'
resume_from = './work_dirs/yolox_x_14-3-23/epoch_500.pth'