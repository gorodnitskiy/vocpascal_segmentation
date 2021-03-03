This repo contains multiclass image segmentation for [VOCPascal-2012](http://host.robots.ox.ac.uk/pascal/VOC/) (33 classes + background).  
Using:
 - [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework  
 - LightRefineNet with EfficientNet-b0 (pretrained on the imagenet) as a backbone and CRP blocks  
 - [Albumentations](https://github.com/albumentations-team/albumentations) library for augmentations  
 - IoU as metric and CCE as loss  
