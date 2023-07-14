_base_ = '/home/badboy-002/github/senior_project/bacteria-detection/mmdetection/configs/yolox/bacteria_albu_yolox.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75,),
    neck=dict(
        type='YOLOXPAFPN',      
        in_channels=[192, 384, 768], 
        out_channels=192,
        num_csp_blocks=2),        
    bbox_head=dict(in_channels=192, feat_channels=192),
    
)


runner = dict(type='EpochBasedRunner', max_epochs=1000)
load_from = "./work_dirs/yolox_m_8-12-22/best_bbox_mAP_epoch_720.pth"
work_dir = './work_dirs/yolox_m_1-3-23'
resume_from = './work_dirs/yolox_m_1-3-23/epoch_800.pth'