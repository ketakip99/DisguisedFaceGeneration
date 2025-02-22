#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:15:09 2024

@author: rajs
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from functools import partial
import random
from torch import cat

transforms = { 'sketch': T.Compose([T.Grayscale(), 
                                    T.Resize(512), 
                                    T.ToTensor(),
                                    #T.Normalize((0.5), (0.5)),
                                    ]), 

              'photo': T.Compose([T.Resize(512), 
                                  T.ToTensor(),
                                  #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]),
              }

class Augmentation:
    @staticmethod
    def random_affine(image, p = 1):
        if random.random() > p:
            return image

        param_affine = T.RandomAffine.get_params(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            scale_ranges=(0.95, 1.05),
            shears=(-10, 10),
            img_size=(512, 512)   
        )

        return T.functional.affine(image,param_affine[0], param_affine[1],
                                       param_affine[2], param_affine[3])

    @staticmethod
    def random_erase(image, p = 1):
        image = T.ToTensor()(image)
        image = T.RandomErasing(p = p, value = 1)(image)
        image = T.ToPILImage()(image)
        return image


class S2F_Dataset(Dataset):    
    def __init__(self, path, transforms = None, load_photo = True, 
                 augmentation=False):
        
        super().__init__()
        
        self.load_photo = load_photo
        self.augmentation = Augmentation() if augmentation else None
        
        self.transforms = transforms
        
        folder_sketch = os.path.join(path, 'sketches')
        folder_photo = os.path.join(path, 'images')
        
        self.path_sketch = []
        self.path_photo = []
        
        for file_name in os.listdir(folder_sketch):
            if file_name.endswith('_sketch.png'):
                sketch_path = os.path.join(folder_sketch, file_name)
                self.path_sketch.append(sketch_path)
                
                if self.load_photo:
                    photo_file_name = file_name.replace('_sketch.png', '_crop.png')
                    photo_path = os.path.join(folder_photo, photo_file_name)
                    
                    if os.path.exists(photo_path):
                        self.path_photo.append(photo_path)
                    else:
                        print(f"Warning: Photo file {photo_path} does not exist.")
    
    def __len__(self):
        return len(self.path_sketch)
            
    def __getitem__(self, idx):        
        assert idx < len(self), 'Index is out of range'
        
        if self.load_photo:
            return self.load_pair(idx)
        else:
            return partial(self.load_one_, 'sketch')(idx)

    def load_one_(self, part, idx):
        path = eval(f'self.path_{part}[idx]')
        pic = Image.open(path)
        if self.augmentation:
            if part == 'sketch':
                pic = self.augmentation.random_erase(pic, p = 1.0)
            pic = self.augmentation.random_affine(pic)
        
        if self.transforms is not None:
            pic = self.transforms[part](pic)
            
        return pic
    
    
    def load_pair(self, idx):
        return [partial(self.load_one_, part)(idx) for part in ['sketch', 'photo']]
        
    

def imshow(sketch, photo = None):    
    sketch = T.ToPILImage()(sketch)
    sketch.show()
    if photo is not None:
        photo = T.ToPILImage()(photo)
        photo.show()

 
        
def save(sketch, photo, generated, name):          
    sketch = sketch.repeat(3,1,1)
    cat_imgs = cat((sketch, photo, generated), dim = 2)
    cat_imgs = T.ToPILImage()(cat_imgs)
    cat_imgs.save(name)
    
    
        
def dataloader(path, batch_size, load_photo=True, shuffle=True, 
               augmentation=False, num_workers=4):
    
    global transforms
    custom_dataset = S2F_Dataset(path,  transforms = transforms,
                                  load_photo = load_photo, augmentation = augmentation)
    custom_dataloader = DataLoader(custom_dataset, batch_size = batch_size, 
                                   shuffle = shuffle, num_workers = num_workers)
    return custom_dataloader


if __name__ == '__main__':
    # tf = Augmentation()
    # path = '../val/sketches'
    # for file_name in os.listdir(path):
    #     print(file_name)
    #     sketch = Image.open(os.path.join(path, file_name))        
    #     sk1 = tf.random_affine(sketch, p = 0.5)
    #     sk2 = tf.random_erase(sk1, p = 0.9)
        
    #     sketch.show()
    #     sk1.show()
    #     sk2.show()
    #     break
    
    path = '/home/user/sk2df/source'
    load_photo = False
    batch_size = 5

    # c = S2F_Dataset(path, transforms = transforms, load_photo = True)
    # idx = random.randint(0, len(c))
    # imshow(*c[idx])
    
    d = dataloader(path, batch_size = batch_size, load_photo = load_photo)
    
    from tqdm import tqdm
    index = 0
    for inputs in tqdm(d, desc=f'Epoch - {index + 1} / {batch_size}'):
        if isinstance(inputs, list):
            s, p = inputs
            for pair in zip(s, p):
                imshow(*pair)
        else:
            for sketch in inputs:                
                imshow(sketch)
        index+=1
        break
    
    
