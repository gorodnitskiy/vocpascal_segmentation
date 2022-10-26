from typing import Optional
import os
import urllib
import tarfile
import torch
import numpy as np
from PIL import Image
import albumentations as A


class VOCdataset:
    def __init__(
        self,
        split: str = 'train',
        root: str = 'datasets',
        aug: Optional[A.Compose] = None
    ) -> None:
        self.root = root
        self.aug = aug
        tar_path = os.path.join(root, 'VOC/VOC.zip')
        dir_path = os.path.join(root, 'VOC')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            with open(tar_path, 'wb+') as file:
                file.write(urllib.request.urlopen(url).read())
                file.close()

        if os.path.exists(tar_path):
            with tarfile.open(tar_path) as tar_ref:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar_ref, dir_path)

            os.remove(tar_path)

        self.images = []
        for line in open(os.path.join(
                dir_path, 'VOCdevkit', 'VOC2012', 'ImageSets',
                'Segmentation', split + '.txt')):
            self.images.append(line.strip())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(
            self.root, 'VOC', 'VOCdevkit', 'VOC2012',
            'JPEGImages', img_name + '.jpg')
        seg_path = os.path.join(
            self.root, 'VOC', 'VOCdevkit', 'VOC2012',
            'SegmentationClass', img_name + '.png')

        img = np.array(Image.open(img_path))
        seg = np.array(Image.open(seg_path))
        res = self.aug(image=img, mask=seg)

        img = torch.tensor(res['image']).permute(2, 0, 1)
        seg = torch.tensor(res['mask'])
        seg[seg == 255] = 0

        return {'image': img.float() / 255.0, 'seg': seg.long()}
