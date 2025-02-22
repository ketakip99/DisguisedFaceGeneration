

from torch import device, set_grad_enabled, sigmoid
from torch.cuda import is_available
from torch.optim import Adam

import dataset, losses
from tqdm import tqdm
import new_components as components

batch_size = 1

load_photo = True

train_path = '/home/asds/sk2df/source'
val_path = '/home/asds/sk2df/source'
test_path = '/home/asds/sk2df/source/testing samples/'

modes = ['train', 'val', 'test']
dataset_paths = {'train': train_path, 'val': val_path, 'test': test_path}

device_name = device('cuda:0') if is_available() else device('cpu')
print(device_name)

spatial_channels = 1
image_channels = 3

gan_kwargs = {'generator' : {'encoder_channels_list' : [spatial_channels, 64, 128, 256],
                            'residual_layers' : 9,
                            'decoder_channels_list' : [256, 128, 64, image_channels]},

              'discriminator' : {'disc_channels_list' : [image_channels, 64, 128, 256, 512, 1],
                               'pool': 0,
                               }
              }

gan = components.Cycle_GAN(**gan_kwargs)

gan.load_model()

gan.to(device_name)

gan.eval()

dataloaders = {x : dataset.dataloader(dataset_paths[x], 
                                      batch_size = batch_size, 
                                      load_photo = load_photo, augmentation=False)
                   for x in modes}

print(f'''Len of datasets --> 
      train: {len(dataloaders["train"])} 
      val: {len(dataloaders["val"])}
      test: {len(dataloaders["test"])}
      ''')

for x in modes:    
    for idx, (sketches, images) in enumerate(dataloaders[x]):
        spatial_map = sketches.to(device_name)
        
        with set_grad_enabled(gan.training):
            fake = gan(spatial_map)[0]
        
        dataset.imshow(sketches[0], fake)
        dataset.save(sketches[0], images[0], fake.cpu(), f'{idx}.jpg')
        break



    
    
    
    
    
    
    
    
    
