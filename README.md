# pytorch-pretrained-gluonresnet

A stand-alone version of the pretrained MxNet Gluon ResNet models ported to Pytorch. Currently part of my model collection (https://github.com/rwightman/pytorch-image-models/blob/master/models/gluon_resnet.py)

This model covers all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet found in the gluon model zoo (https://gluon-cv.mxnet.io/model_zoo/classification.html, https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/model_zoo.py) that
  * have stride in 3x3 conv layer of bottleneck
  * have conv-bn-act ordering

Included ResNet variants are:
  * v1b - 7x7 stem, stem_width=64, same as torchvision ResNet (checkpoint compatible), or NVIDIA ResNet 'v1.5'
  * v1c - 3 layer deep 3x3 stem, stem_width = 32
  * v1d - 3 layer deep 3x3 stem, stem_width = 32, average pool in downsample
  * v1e - 3 layer deep 3x3 stem, stem_width = 64, average pool in downsample  *no pretrained weights available
  * v1s - 3 layer deep 3x3 stem, stem_width = 64

ResNeXt is standard and checkpoint compatible with torchvision pretrained models. 7x7 stem,
    stem_width = 64, standard cardinality and base width calcs

SE-ResNeXt is standard. 7x7 stem, stem_width = 64,
    checkpoints are not compatible with Cadene pretrained, but could be with key mapping

SENet-154 is standard. 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
    reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

Original ResNet-V1, ResNet-V2 (bn-act-conv), and SE-ResNet (stride in first bottleneck conv) are NOT supported.


## Results
|                    |                 |                 |             | 
|--------------------|-----------------|-----------------|-------------| 
| model              | top1 (err)      | top5 (err)      | param_count | 
| resnet18_v1b       | 70.830 (29.170) | 89.756 (10.244) | 11.69       | 
| resnet34_v1b       | 74.580 (25.420) | 91.988 (8.012)  | 21.8        | 
| resnet50_v1b       | 77.578 (22.422) | 93.718 (6.282)  | 25.56       | 
| resnet50_v1c       | 78.010 (21.990) | 93.988 (6.012)  | 25.58       | 
| resnet50_v1s       | 78.712 (21.288) | 94.242 (5.758)  | 25.68       | 
| resnet50_v1d       | 79.074 (20.926) | 94.476 (5.524)  | 25.58       | 
| resnet101_v1b      | 79.304 (20.696) | 94.524 (5.476)  | 44.55       | 
| resnext50_32x4d    | 79.356 (20.644) | 94.424 (5.576)  | 25.03       | 
| resnet101_v1c      | 79.544 (20.456) | 94.586 (5.414)  | 44.57       | 
| resnet152_v1b      | 79.692 (20.308) | 94.738 (5.262)  | 60.19       | 
| seresnext50_32x4d  | 79.912 (20.088) | 94.818 (5.182)  | 27.56       | 
| resnet152_v1c      | 79.916 (20.084) | 94.842 (5.158)  | 60.21       | 
| resnet101_v1s      | 80.300 (19.700) | 95.150 (4.850)  | 44.67       | 
| resnext101_32x4d   | 80.334 (19.666) | 94.926 (5.074)  | 44.18       | 
| resnet101_v1d      | 80.424 (19.576) | 95.020 (4.980)  | 44.57       | 
| resnet152_v1d      | 80.470 (19.530) | 95.206 (4.794)  | 60.21       | 
| resnext101_64x4d   | 80.602 (19.398) | 94.994 (5.006)  | 83.46       | 
| seresnext101_64x4d | 80.890 (19.110) | 95.304 (4.696)  | 88.23       | 
| seresnext101_32x4d | 80.902 (19.098) | 95.294 (4.706)  | 48.96       | 
| resnet152_v1s      | 81.012 (18.988) | 95.416 (4.584)  | 60.32       | 
| senet154           | 81.224 (18.776) | 95.356 (4.644)  | 115.09      | 


## PyTorch Hub

Models can be access via the PyTorch Hub API

```
>>> torch.hub.list('rwightman/pytorch-pretrained-gluonresnet')
['gluon_resnet18_v1b', ...]
>>> model = torch.hub.load('rwightman/pytorch-pretrained-gluonresnet', 'gluon_resnet50_v1d', pretrained=True)
>>> model.eval()
>>> output = model(torch.randn(1,3,224,224))
```