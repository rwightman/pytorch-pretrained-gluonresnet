from gluon_resnet import *
from utils import load_checkpoint


def model_names():
    return gluon_resnet_model_names


_model_fns = {n: globals()[n] for n in model_names()}


def create_model(
        model_name='mnasnet_100',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):

    margs = dict(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)

    if model_name in _model_fns:
        create_fn = _model_fns[model_name]
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
