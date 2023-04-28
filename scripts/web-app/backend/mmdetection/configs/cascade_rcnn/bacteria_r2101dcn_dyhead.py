_base_ = ['./bacteria_r2101_dyhead.py']

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)

# runtime settings
evaluation = dict(save_best='auto',
    interval=1, 
    metric='bbox',
    # ap each class
    classwise=True)

runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=10)
load_from = "./work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/best_bbox_mAP_epoch_16.pth"
# resume_from = "./work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/best_bbox_mAP_epoch_16.pth"
work_dir = './work_dirs/crcnn_r2101_dcnv2_8class_24-2-23'
