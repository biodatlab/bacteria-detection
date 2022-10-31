_base_ = './bacteria_centernet_resnet18.py'
model = dict(neck=dict(use_dcn=False))
