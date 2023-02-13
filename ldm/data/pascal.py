import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PascalBase(Dataset):
    CLASS_NAMES_LIST = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor"
    ]
    
    def __init__(self, root=None, train=False, size=128, interpolation='bicubic', SCALING=20):
        self.train = train
        
        base_dir = os.path.join(root, "PASCAL_SBDAUG")
        assert os.path.exists(base_dir)

        if train:
            name_path = os.path.join(base_dir, 'trainaug.txt')
            class_map_dir = os.path.join(base_dir, 'trainaug_class_map')
        else:
            name_path = os.path.join(base_dir, 'val.txt')
            class_map_dir = os.path.join(base_dir, 'val_class_map')
        
        # Read files
        name_list = list(np.loadtxt(name_path, dtype='str'))
        self.images = [os.path.join(base_dir, "raw_images", n + ".jpg") for n in name_list]
        self.targets = [os.path.join(base_dir, "annotations", n + ".npy") for n in name_list]
        
        # Given a class idx (1-20), self.class_map gives the list of images that contain
        # this class idx
        self.class_map = {}
        for c in range(1, 21):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            class_name_list = list(np.loadtxt(class_map_path, dtype='str'))
            # Map name to indices
            class_idx_list = [name_list.index(n) for n in class_name_list]
            self.class_map[c] = class_idx_list
        
        assert size is not None, "size cutting in __getitem__ requires size to be not None"

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=0.5)

        self.SCALING=SCALING

    def __len__(self):
        return int(len(self.images) * self.SCALING)

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        example = {}
        
        example['file_path_'] = self.images[idx]
        example['class_label'] = 1000
        example['human_label'] = 'unassigned'
        # For both SBD and VOC2012, images are stored as .jpg
        img = Image.open(example['file_path_'])
        
        if not img.mode == 'RGB':
            img = img.convert('RGB')

        # Use score-sde preprocessing like LSUN in original latent diffusion
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
    
        target_np = np.load(self.targets[idx])
        if self.size is not None:
            target_mask = Image.fromarray(target_np)
            target_mask = target_mask.resize((self.size, self.size), resample=PIL.Image.Resampling.NEAREST)
            target_np = np.array(target_mask)
        
        example['target_seg_mask'] = torch.tensor(target_np).long()

        return example

class PascalFullTrain(PascalBase):
    def __init__(self, **kwargs):
        super().__init__(root='/data', train=True, **kwargs)

class PascalFullVal(PascalBase):
    def __init__(self, **kwargs):
        super().__init__(root='/data', train=False, **kwargs)