""" PyTorch hubconf.py

## Users can get this published model by calling:
hub_model = hub.load(
    'rwightman/pytorch-pretraiend-gluonresnet:master', # repo_owner/repo_name:branch
    'gluon_resnet50_v1d', # entrypoint
    pretrained=True) # kwargs for callable
"""
dependencies = ['torch', 'math']

from gluon_resnet import gluon_resnet18_v1b, gluon_resnet34_v1b, gluon_resnet50_v1b, gluon_resnet101_v1b,\
    gluon_resnet152_v1b, gluon_resnet50_v1c, gluon_resnet101_v1c, gluon_resnet152_v1c, gluon_resnet50_v1d,\
    gluon_resnet101_v1d, gluon_resnet152_v1d, gluon_resnet50_v1e, gluon_resnet101_v1e, gluon_resnet152_v1e,\
    gluon_resnet50_v1s, gluon_resnet101_v1s, gluon_resnet152_v1s, gluon_resnext50_32x4d,\
    gluon_resnext101_32x4d, gluon_resnext101_64x4d, gluon_resnext152_32x4d, gluon_seresnext50_32x4d,\
    gluon_seresnext101_32x4d, gluon_seresnext101_64x4d, gluon_seresnext152_32x4d, gluon_senet154
