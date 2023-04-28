_base_ = ['./bacteria_r2101_dyhead.py']

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)

optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step= [100,150,200,250])
# runtime settings
evaluation = dict(save_best='auto',
    interval=1, 
    metric='bbox',
    # ap each class
    classwise=True)

runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=5)
# load_from = "./work_dirs/crcnn_r2101_dyhead_7-2-23/best_bbox_mAP_epoch_110.pth"
# resume_from = "./work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/best_bbox_mAP_epoch_16.pth"
work_dir = './work_dirs/crcnn_r2101_dcnv2_pretrain_19-2-23'
load_from = "./work_dirs/cascade_r50_pretrain_19-2-23/epoch_10.pth"
resume_from = "./work_dirs/crcnn_r2101_dcnv2_pretrain_19-2-23/epoch_5.pth"