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
