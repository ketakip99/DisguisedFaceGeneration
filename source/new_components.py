#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:57:51 2024

@author: rajs
"""

from new_common import constants, Encoder, Decoder, ConvBlock, ConvTransposeBlock, ResNetBlock, ResidualBlock
from torch.nn import Identity, Module, ModuleDict, Sequential, AvgPool2d, ReLU, AdaptiveAvgPool2d, Tanh, LeakyReLU
from torch import sigmoid, save, load, device, cat
from os import path, makedirs
# import multiprocessing
from functools import partial
from collections import OrderedDict


class Base(Module):    
    saved_model_path = './runs'
    saved_model_extn = '.pt'
    def __init__(self):
        super().__init__()


    def __msave__(self, model, name):
        if not path.exists(self.saved_model_path):
            makedirs(self.saved_model_path)

        if not isinstance(model, Identity):
            save(model.state_dict(),
                 path.join(self.saved_model_path, name) + self.saved_model_extn
                 )

    
    def __mload__(self, model, name, map_location):
        name = path.join(self.saved_model_path, name) + self.saved_model_extn
        
        if path.exists(name):
            print(f'Loading {name}', end = '... ')        
            model.load_state_dict(load(name, 
                                       map_location = map_location, 
                                       # weights_only=True
                                       )
                                  )
            print(f'Loaded {name}')


class Autoencoder(Base):
    def __init__(self, encoder=None, decoder=None, dec_output_dimension = 512, 
                 resnet_f = 1, **kwargs):
        super().__init__()

        encoder_channels_list = kwargs['encoder_channels_list']
        decoder_channels_list = kwargs['decoder_channels_list']

        self.ae_layers = {
            'encoder': encoder_channels_list,
            'decoder': decoder_channels_list,
        }
               
        if encoder:
            self.encoder = Encoder(self.ae_layers['encoder'], 
                                   resnet_f=resnet_f,
                                   input_dimension =  dec_output_dimension)
        else:
            self.encoder = self.skip

        if decoder:
            self.decoder = Decoder(self.ae_layers['decoder'], 
                                   encoder_stride = 2**(len(self.ae_layers['encoder']) - 2),
                                   output_dimension = dec_output_dimension, 
                                   resnet_f=resnet_f)
        else:
            self.decoder = self.skip

    def forward(self, _x):
        latent = self.encoder(_x)        
        return self.decoder(latent)
        

    def save_(self, part, model_name):                
        self.__msave__(eval(f'self.{part}'), model_name)

    def load_(self, part, model_name, map_location):                
        self.__mload__(eval(f'self.{part}'), model_name, map_location)


class AutoencoderPool(Base):
    components = ['LtEye', 'RtEye', 'Nose', 'Mouth', 'Background']
    component_rois = {
        components[0]: [108, 126, 128, 128], # [x, y, w, h]
        components[1]: [255, 126, 128, 128],  # [x, y, w, h]
        components[2]: [182, 232, 160, 160],  # [x, y, w, h]
        components[3]: [169, 301, 192, 192],  # [x, y, w, h]
        components[4]: [0, 0, 512, 512],  # [x, y, w, h]
    }    
    
    def __init__(self, **kwargs):
        super().__init__()

        # self.comp_ae_dict = ModuleDict({comp: Autoencoder(encoder=True, decoder=True,
        #                                        dec_output_dimension = self.component_rois[comp][-1],
        #                                        **kwargs
        #                                  )
        #                for comp in self.components
        #                })

        self.comp_ae_dict = {comp: Autoencoder(encoder=True, decoder=True,
                                               dec_output_dimension = self.component_rois[comp][-1],
                                               **kwargs
                                         )
                       for comp in self.components
                       }

    def split(self, sketch):
        patches = {}
        for key, roi in self.component_rois.items():
            x1, y1, w, h = roi
            x2, y2 = x1 + w, y1 + h
            patches[key] = sketch[:, :, y1:y2, x1:x2].clone()
        return patches


    def forward_pass(self, patches):        
        rec_patches = {}        
        for key, patch in patches.items():
            rec_patches[key] = self.comp_ae_dict[key](patch)
        return rec_patches
    
    
    def save_all_(self, part):
        model_names = eval(f'self.{part}_saved_models_name')
        for key, model_name in model_names.items():
            partial(self.comp_ae_dict[key].save_, part)(model_name)


    def load_all_(self, part, map_location):
        model_names = eval(f'self.{part}_saved_models_name')
        for key, model_name in model_names.items():
            partial(self.comp_ae_dict[key].load_, part)(model_name, map_location)
            
    def save_models(self, encoder = True, decoder = True):
        if encoder:
            partial(self.save_all_, 'encoder')()

        if decoder:
            partial(self.save_all_, 'decoder')()


    def load_models(self, encoder = True, decoder = True, map_location = device('cpu')):
        if encoder:
            partial(self.load_all_, 'encoder')(map_location)

        if decoder:
            partial(self.load_all_, 'decoder')(map_location)            

            
class Components(AutoencoderPool):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_saved_models_name = { comp: f'{comp}_encoder' 
                                            for comp in self.components}
        self.decoder_saved_models_name = { comp: f'CE_{comp}_decoder' 
                                            for comp in self.components}

    def forward(self, sketch):
        rec_sketches = self.forward_pass(sketch)
        # print(type(rec_sketches))
        return {key: sigmoid(rec_sketch) for key, rec_sketch in rec_sketches.items()}
            

class FeatureMapping(AutoencoderPool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder_saved_models_name = { comp: f'{comp}_encoder' 
                                            for comp in self.components}

        self.decoder_saved_models_name = { comp: f'FM_{comp}_decoder' 
                                            for comp in self.components}

        self.load_models(encoder = True, decoder = True)

    def forward(self, sketch):
        patches = self.forward_pass(sketch)
        spatial_map = self.merge(patches)        
        return spatial_map

    def merge(self, patches):
        spatial_map = patches[self.components[-1]].clone()
        for key in self.components[:-1]:
            x1, y1, w, h = self.component_rois[key]
            x2, y2 = x1 + w, y1 + h            
            spatial_map[:, :, y1:y2, x1:x2] = patches[key]
        return spatial_map
            

class Generator(Module):
    def __init__(self, **kwargs):
        super().__init__()
        encoder_channels_list = kwargs['encoder_channels_list'] #[32, 56, 112, 224, 448]
        resnet_layers = kwargs['resnet_layers']
        decoder_channels_list = kwargs['decoder_channels_list'] #[448, 224, 112, 56, 3]
        
        layers = OrderedDict()
        n = 1
        for ci, co in zip(encoder_channels_list[:-1], encoder_channels_list[1:]):
            layers[f'conv{n}'] = ConvBlock(ci, co, norm_type = constants.NO_NORM)
            n+=1

        n = 1
        for i in range(resnet_layers):
            layers[f'resnet{n}'] = ResNetBlock(encoder_channels_list[-1], f=1)
            n+=1

        n = 1
        for ci, co in zip(decoder_channels_list[:-2], decoder_channels_list[1:-1]):
            layers[f'convT{n}'] = ConvTransposeBlock(ci, co, norm_type = constants.NO_NORM)
            n += 1

        layers[f'convT{n}'] = ConvTransposeBlock(decoder_channels_list[-2], 
                                                 decoder_channels_list[-1], 
                                                 activation = False, 
                                                 norm_type = constants.NO_NORM)
        
        self.layers = Sequential(layers)
        
    def forward(self, _x):
        return self.layers(_x)
    

class Discriminator(Module):
    def __init__(self,**kwargs):
        super().__init__()
        disc_channels_list = kwargs['disc_channels_list']

        layers = OrderedDict()
        n = 1
        for ci, co in zip(disc_channels_list[:-2], disc_channels_list[1:-1]):
            layers[f'conv{n}'] = ConvBlock(ci, co, norm_type = constants.NO_NORM)
            n += 1            
        layers[f'conv{n}'] = ConvBlock(disc_channels_list[-2], 
                                       disc_channels_list[-1], 
                                       activation = False, 
                                       norm_type = constants.NO_NORM)                   
        self.layers = Sequential(layers)

    def forward(self, _x):
        return self.layers(_x)


class GAN(Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.components = ['generator', 'discriminator']
        
        gen_kwargs = kwargs[self.components[0]]
        dis_kwargs = kwargs[self.components[1]]        
        self.generator = Generator(**gen_kwargs)
        self.discriminator = Discriminator(**dis_kwargs)
                
        self.num_pool = dis_kwargs['pool']
        self.pool = AvgPool2d(kernel_size=4, stride=2, padding=1)        
    
    def forward(self, _x):        
        return self.generator(_x)
            
    def discriminate(self, x1, x2):
        x = cat((x1, x2), dim = 1)

        discs = [self.discriminator(x)]
        for i in range(1, self.num_pool):
            x = self.pool(x)
            discs.append(self.discriminator(x))       
        
        return discs

        
    def save_(self, part):
        self.__msave__(eval(f'self.{part}'), part)

    def load_(self, part, map_location = device('cpu')):                
        self.__mload__(eval(f'self.{part}'), part, map_location)
    
    def save_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.save_, self.components[0])()

        if discriminator:
            partial(self.save_, self.components[1])()


    def load_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.load_, self.components[0])()
        
        if discriminator:
            partial(self.load_, self.components[1])()


class GAN2(GAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dis_kwargs = kwargs[self.components[1]]        
        self.discriminator1 = Discriminator(**dis_kwargs)
        self.discriminator2 = Discriminator(**dis_kwargs)
                        
    def discriminate(self, x1, x2):
        x = cat((x1, x2), dim = 1)

        discs = [self.discriminator(x)]
        for i in range(1, self.num_pool):
            x = self.pool(x)
            discs.append(eval(f'self.discriminator{i}(x)'))     
        
        return discs


class P2P_Generator(Module):
    def __init__(self, **kwargs):
        super().__init__()
        encoder_channels_list = kwargs['encoder_channels_list'] #[32, 56, 112, 224, 448]
        decoder_channels_list = kwargs['decoder_channels_list'] #[448, 224, 112, 56, 3]
        
        p2p_act = ReLU()
        layers = OrderedDict()
        n = 1
        for ci, co in zip(encoder_channels_list[:-2], encoder_channels_list[1:-1]):
            layers[f'conv{n}'] = ConvBlock(ci, co, activation = p2p_act, norm_type = constants.NO_NORM)
            n+=1

        layers[f'conv{n}'] = ConvBlock(encoder_channels_list[-2], encoder_channels_list[-1], stride = 1, activation = p2p_act, norm_type = constants.NO_NORM)

        n = 1
        layers[f'convT{n}'] = ConvTransposeBlock(decoder_channels_list[0], decoder_channels_list[1], stride = 1, activation = p2p_act, norm_type = constants.NO_NORM)
        n+=1
        for ci, co in zip(decoder_channels_list[1:-2], decoder_channels_list[2:-1]):
            layers[f'convT{n}'] = ConvTransposeBlock(ci, co, activation = p2p_act, norm_type = constants.NO_NORM)
            n += 1

        layers[f'convT{n}'] = ConvTransposeBlock(decoder_channels_list[-2], decoder_channels_list[-1], activation = False, norm_type = constants.NO_NORM)
        
        self.layers = Sequential(layers)
        
    def forward(self, _x):
        return self.layers(_x)
        

class P2P_Discriminator(Module):
    def __init__(self, **kwargs):
        super().__init__()
        disc_channels_list = kwargs['disc_channels_list']
        
        p2p_act = ReLU()
        layers = OrderedDict()
        n = 1
        for ci, co in zip(disc_channels_list[:-2], disc_channels_list[1:-1]):
            layers[f'conv{n}'] = ConvBlock(ci, co, activation = p2p_act, norm_type = constants.NO_NORM)
            n += 1            

        layers[f'conv{n}'] = ConvBlock(disc_channels_list[-2], disc_channels_list[-1], stride = 1, activation = False, norm_type = constants.NO_NORM)              
        self.layers = Sequential(layers)
        
        self.pool = AdaptiveAvgPool2d(1)

    def forward(self, _x):
        return self.pool(self.layers(_x))


class P2P_GAN(Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.components = ['generator', 'discriminator']
        
        gen_kwargs = kwargs[self.components[0]]
        dis_kwargs = kwargs[self.components[1]]        
        self.generator = P2P_Generator(**gen_kwargs)
        self.discriminator = P2P_Discriminator(**dis_kwargs)                
    
    def forward(self, _x):        
        return self.generator(_x)
            
    def discriminate(self, x1, x2):
        x = cat((x1, x2), dim = 1)
        return self.discriminator(x)
        
    def save_(self, part):
        self.__msave__(eval(f'self.{part}'), 'p2p_' + part)

    def load_(self, part, map_location = device('cpu')):                
        self.__mload__(eval(f'self.{part}'), 'p2p_' + part, map_location)
    
    def save_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.save_, self.components[0])()

        if discriminator:
            partial(self.save_, self.components[1])()


    def load_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.load_, self.components[0])()
        
        if discriminator:
            partial(self.load_, self.components[1])()
            
            
            
#-------------------------------------------------------------------------------------

class Cycle_Generator(Module):
    def __init__(self, **kwargs):
        super().__init__()
        encoder_channels_list = kwargs['encoder_channels_list']
        residual_layers = kwargs['residual_layers'] 
        decoder_channels_list = kwargs['decoder_channels_list'] 
        
        final_act = Tanh()
        cycle_act = ReLU()
        layers = OrderedDict()
        
        n = 1
        layers[f'conv{n}'] = ConvBlock(encoder_channels_list[0], encoder_channels_list[1], stride = 1, activation = cycle_act, norm_type = constants.INSTANCE_NORM, padding =3, padding_mode = "reflect", kernel_size = 7)
        n+=1
        for ci, co in zip(encoder_channels_list[1:-1], encoder_channels_list[2:]):
            layers[f'conv{n}'] = ConvBlock(ci, co, activation = cycle_act, norm_type = constants.INSTANCE_NORM, padding = 1, stride = 2, kernel_size = 3)
            n+=1   
        n = 1
        for i in range(residual_layers):
            layers[f'residual{n}'] = ResidualBlock(encoder_channels_list[-1])
            n+=1
        n = 1
        for ci, co in zip(decoder_channels_list[:-2], decoder_channels_list[1:-1]):
            layers[f'convT{n}'] = ConvTransposeBlock(ci, co, activation = cycle_act, kernel_size = 3, stride = 2, padding = 1, norm_type = constants.INSTANCE_NORM, output_padding = 1) 
            n += 1
        n = 4
        layers[f'conv{n}'] = ConvBlock(decoder_channels_list[-2], decoder_channels_list[-1], activation = final_act, norm_type = constants.NO_NORM, stride = 1, padding = 3, padding_mode = "reflect", kernel_size = 7)
        
        
        self.layers = Sequential(layers)
        
    def forward(self, _x):
        return self.layers(_x)
        
        
        

class Cycle_Discriminator(Module):
    def __init__(self, **kwargs):
        super().__init__()
        disc_channels_list = kwargs['disc_channels_list']
        
        dis_act = LeakyReLU(0.2)
        layers = OrderedDict()
        n = 1
        for ci, co in zip(disc_channels_list[:-3], disc_channels_list[1:-2]):
            layers[f'conv{n}'] = ConvBlock(ci, co, activation = dis_act, norm_type = constants.INSTANCE_NORM, kernel_size = 4, stride = 2, padding = 1)
            n += 1            

        layers[f'conv{n}'] = ConvBlock(disc_channels_list[-3], disc_channels_list[-2], stride = 1, activation = dis_act, norm_type = constants.INSTANCE_NORM,  kernel_size = 4, padding = 1)
        
        n+=1
        layers[f'conv{n}'] = ConvBlock(disc_channels_list[-2], disc_channels_list[-1],activation = False, norm_type = constants.INSTANCE_NORM,  kernel_size = 4, padding = 1)     
        
        
                 
        self.layers = Sequential(layers)

    def forward(self, _x):
        return self.layers(_x)


class Cycle_GAN(Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.components = ['generator', 'discriminator']
        
        gen_kwargs = kwargs[self.components[0]]
        dis_kwargs = kwargs[self.components[1]]        
        self.generator = Cycle_Generator(**gen_kwargs)
        self.discriminator = Cycle_Discriminator(**dis_kwargs)                
    
    def forward(self, _x):        
        return self.generator(_x)
            
    def discriminate(self, x):
        #x = cat((x1, x2), dim = 1)
        return self.discriminator(x)
        
    def save_(self, part):
        self.__msave__(eval(f'self.{part}'), 'cycle_' + part)

    def load_(self, part, map_location = device('cpu')):                
        self.__mload__(eval(f'self.{part}'), 'cycle_' + part, map_location)
    
    def save_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.save_, self.components[0])()

        if discriminator:
            partial(self.save_, self.components[1])()


    def load_model(self, generator= True, discriminator = True):
        if generator:
            partial(self.load_, self.components[0])()
        
        if discriminator:
            partial(self.load_, self.components[1])()
                                        
    
if __name__ == "__main__":
    sk_channels = 1
    input_dim = 512
    from torch import rand
    x = rand(5, sk_channels, input_dim, input_dim)
    print(x.shape)

    # c = Autoencoder(encoder=True, decoder=True, de_output_dimension=input_dim,
    #                 ae_kwargs = {'input_channels': sk_channels,
    #                  'num_layers': 5,
    #                  'latent_channels' : 512 },
    #                 )
    # print(c.ae_layers)
    # y = c(x)
    # print(y.shape)
    # c.save_model()
    #
    # c2 = Autoencoder(encoder=True, decoder=False, de_output_dimension=input_dim,
    #                 ae_kwargs={'input_channels': sk_channels,
    #                            'num_layers': 5,
    #                            'latent_channels': 512},
    #                 )
    # c2.load_model()

    c_kwargs = {'encoder_channels_list': [1, 32, 64, 128, 256, 512, 512],
                 'decoder_channels_list': [512, 512, 256, 128, 64, 32, 32, 1],
                 }
    c = Components(**c_kwargs)
    p = c.split(x)
    rec_x = c(p)

    for k, r in rec_x.items():
        print('CE Output: ', k, r.shape)
        
    # c.save_models(encoder = True, decoder = True)
    
    # spatial_channels = 32
    # image_channels = 3
    # f_kwargs = {'encoder_channels_list': [1, 32, 64, 128, 256, 512, 512],
    #               'decoder_channels_list': [512, 512, 256, 256, 128, 64, 64, spatial_channels]}
    # f = FeatureMapping(**f_kwargs)
    # p = f.split(x)
    # fm_x = f(p)
    # print('FeatureMapping Output: ', fm_x.shape)
    
    # f.save_models(encoder = False, decoder = True)
    
    
    # # for p1, p2 in zip(c.comp_ae_dict[c.components[0]].named_parameters(), f.comp_ae_dict[f.components[0]].named_parameters()):
    # #     if 'encoder' in p1[0]:
    # #         print(p1[0], p2[0])
    # #         print((p1[1] == p2[1]).all())
    
    # gan_kwargs = {'generator' : {'encoder_channels_list' : [spatial_channels, 56, 112, 224, 448],
    #                             'resnet_layers' : 9,
    #                             'decoder_channels_list' : [448, 224, 112, 56, image_channels]},
    
    #               'discriminator' : {'disc_channels_list' : [spatial_channels + image_channels, 64, 128, 256, 512],
    #                                'pool': 3,
    #                                }
    #               }
    
    # gan = GAN(**gan_kwargs)
    
    # gan.save_model()
    
    # gen = gan(fm_x)
    # print('Gen Output: ', gen.shape)
    
    # dis = gan.discriminate(fm_x, gen)
    # for i, d in enumerate(dis):
    #     print('Disc Output:', i, d.shape)

    # test_gan = GAN(**gan_kwargs)
    # test_gan.load_model(generator= True, discriminator=True)
    
    # for p1, p2 in zip(gan.named_parameters(), test_gan.named_parameters()):        
    #     print(p1[0], p2[0])
    #     print((p1[1] == p2[1]).all())
    
