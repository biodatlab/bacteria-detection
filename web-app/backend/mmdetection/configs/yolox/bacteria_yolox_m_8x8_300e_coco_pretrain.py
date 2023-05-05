_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/yolox/bacteria_albu_yolox_pretrain.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192),
)

evaluation = dict(
    save_best="auto",
    interval=1,
    metric="bbox",
    # ap each class
    classwise=True,
)

runner = dict(type="EpochBasedRunner", max_epochs=300)
checkpoint_config = dict(interval=10)
work_dir = "./work_dirs/yolox_m_pretrain_22-2-23"