import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderReaderBase(Dataset):
    def __init__(self, root=None, size=128, interpolation='bicubic', SCALING=10):
        assert root is not None
        self.root = root
        # Read files
        class_idx_list = os.listdir(root)
        
        self.img_path_list = []
        self.label_list = []
        for cls_str in class_idx_list:
            cls_dir = os.path.join(self.root, cls_str)
            img_file_list = os.listdir(cls_dir)
            for img_fn in img_file_list:
                if img_fn.endswith('.jpg'):
                    self.img_path_list.append(os.path.join(cls_dir, img_fn))
                    self.label_list.append(int(cls_str))
        
        assert size is not None, "size cutting in __getitem__ requires size to be not None"

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=0.5)

        self.SCALING=SCALING

    def __len__(self):
        return int(len(self.img_path_list) * self.SCALING)

    def __getitem__(self, idx):
        idx = idx % len(self.img_path_list)
        example = {}
        
        example['file_path_'] = self.img_path_list[idx]
        example['class_label'] = self.label_list[idx]
        example['human_label'] = 'unassigned'
        # For both SBD and VOC2012, images are stored as .jpg
        img = Image.open(example['file_path_'])
        
        if not img.mode == 'RGB':
            img = img.convert('RGB')

        img = np.array(img).astype(np.uint8)

        # Pad to square by longer edge
        if example['class_label'] != 0:
            if np.random.random() < 1:
                h, w = img.shape[0], img.shape[1]
                if h > w:
                    img = np.pad(img, ((0, 0), ((h - w) // 2, (h - w) // 2), (0, 0)), 'constant')
                elif w > h:
                    img = np.pad(img, (((w - h) // 2, (w - h) // 2), (0, 0), (0, 0)), 'constant')

        # Crop to the shorter edge
        # crop = min(img.shape[0], img.shape[1])
        # h, w, = img.shape[0], img.shape[1]
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #       (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example

class FolderTrain(FolderReaderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FolderVal(FolderReaderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
