import streamlit as st
from torch import device, set_grad_enabled, sigmoid
from torch.cuda import is_available
from torch.optim import Adam

import dataset, losses
from tqdm import tqdm
import new_components as components
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np
from torch.utils.data import DataLoader


#batch_size = 1
#
#load_photo = True
#
#train_path = '/home/user/sk2df/source'
#val_path = '/home/user/sk2df/source'
#test_path = '/home/user/sk2df/240927/source/test/'
#
#modes = ['train', 'val', 'test']
#dataset_paths = {'train': train_path, 'val': val_path, 'test': test_path}

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
#
#transforms = T.Compose([T.Grayscale(), T.Resize(512), T.ToTensor(), ]) 
#
#st.title('Sketch to Image Generator')
#uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"])
#
#if uploaded_file is not None:
#    sketch = Image.open(uploaded_file)
#    sketch_loader = transforms(sketch).to(device_name)
#
#    with set_grad_enabled(gan.training):
#        generated_image = gan(sketch_loader)
#           
#    generated_image = generated_image.cpu().permute(1, 2, 0).numpy()
#    generated_image = np.clip(generated_image, 0, 1)
#    generated_image = (generated_image * 255).astype(np.uint8)
#    
#    
#    col1, col2 = st.columns(2)
#    with col1:
#        st.image(sketch, caption='Uploaded Sketch', use_column_width=True)
#
#    with col2:
#        st.image(generated_image, caption='Generated Image', use_column_width=True)
#
#    
    
    
    
    
    
    
    
    
