#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:57:51 2024

@author: rajs
"""

from torch.nn import Conv2d, ConvTranspose2d, Identity, \
    BatchNorm2d, InstanceNorm2d, LeakyReLU, Module, Sequential, \
    ReLU, ReflectionPad2d, AdaptiveAvgPool2d, Linear, Unflatten
    
from collections import OrderedDict


class constants:
    default_activation = LeakyReLU(0.1)
    skip = Identity()
    NO_NORM = 0
    BATCH_NORM = 1
    INSTANCE_NORM = 2

class base(Module):
    def get_norm_layer(self, c, norm_type):
        if norm_type == constants.NO_NORM:
            return constants.skip
        elif norm_type == constants.BATCH_NORM:
            return BatchNorm2d(c)
        elif norm_type == constants.INSTANCE_NORM:
            return InstanceNorm2d(c)
        else:
            print('Wrong Value Selected.')
            raise ValueError
            
    def forward(self, _x):
        return self.layers(_x)

class ConvBlock(base):

    def __init__(self, in_channels, out_channels,
                 kernel_size = 4, padding = 1, stride = 2,padding_mode='zeros',
                 activation = True, norm_type = 1):
        super(ConvBlock, self).__init__()        

        self.layers = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode=padding_mode, bias = True)),
            ('bn', self.get_norm_layer(out_channels, norm_type)),
            ('act', constants.default_activation if activation==True 
             else activation if isinstance(activation, Module) else constants.skip),
        ]))

class ConvTransposeBlock(base):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 4, padding = 1, output_padding=0, stride = 2, padding_mode='zeros',
                 activation = True, norm_type = 1):
        super(ConvTransposeBlock, self).__init__()

        self.layers = Sequential(OrderedDict([
            ('convT', ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, output_padding= output_padding, stride = stride, padding_mode=padding_mode, bias = True)),
            ('bn', self.get_norm_layer(out_channels, norm_type)),
            ('act', constants.default_activation if activation==True 
             else activation if isinstance(activation, Module) else constants.skip),
        ]))



class ResNetBlock(base):
    def __init__(self, channels, f=1):
        super(ResNetBlock, self).__init__()
        _channels = channels // f
        self.layers = Sequential(OrderedDict([
            ('conv1', ConvBlock(channels, _channels, kernel_size = 3,
                                stride = 1, 
                                activation = ReLU(inplace=True))),
            ('conv2', ConvBlock(_channels, channels, kernel_size = 3, 
                                stride = 1, activation = False, 
                                norm_type = constants.NO_NORM)),
        ]))

    def forward(self, _x):        
        return  self.layers(_x) + _x
        
class ResidualBlock(base):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.layers = Sequential(OrderedDict([
            ('conv1', ConvBlock(channels, channels, kernel_size = 3,
                                stride = 1, padding =1,
                                activation = ReLU(inplace=True), norm_type = constants.INSTANCE_NORM, padding_mode='reflect')),
                                
            ('conv2', ConvBlock(channels, channels, kernel_size = 3, 
                                stride = 1, padding =1, activation = False, 
                                norm_type = constants.INSTANCE_NORM, padding_mode='reflect')),               
        
        
        ]))         

    def forward(self, x):
        return x + self.layers(x)


class Encoder(base):
    def __init__(self, list_channels, resnet_f=2, input_dimension = None):
        super(Encoder, self).__init__()

        hidden_channels = list_channels[:-1]
        output_channel = list_channels[-1]        
        
        layers = OrderedDict()
        for n, (ci, co) in enumerate(zip(hidden_channels[:-1], hidden_channels[1:]), start=1):
            layers[f'conv{n}'] = ConvBlock(ci, co)
            layers[f'resnet{n}'] = ResNetBlock(co, f=resnet_f)
        
        # layers['latent'] = ConvBlock(hidden_channels[-1], output_channel, 
                                      # kernel_size = 1, padding=0, stride = 1, 
                                      # activation = False, norm_type = constants.NO_NORM)        
        # layers['pooling'] = AdaptiveAvgPool2d(1)        
        self.layers = Sequential(layers)
        
        # self.latent = ConvBlock(hidden_channels[-1], output_channel, 
        #                               kernel_size = 1, padding = 0, stride = 1, 
        #                               activation = False, 
        #                               norm_type = constants.NO_NORM)
        # self.pool = nn.AdaptiveAvgPool2d(1)

        fmap_dim = input_dimension//2**len(hidden_channels[1:])
        # print('fmap', fmap_dim)
        self.latent = Linear(hidden_channels[-1] * fmap_dim * fmap_dim, output_channel)

        
    def forward(self, x):
        x = self.layers(x)
        # print('layers x shape', x.shape)
        x = x.view(-1, self.latent.in_features)
        
        # w, h = x.shape[2:]
        # self.latent.layers.conv.stride = (w, h)
        x = self.latent(x)
        return x
        

class Decoder(base):
    @staticmethod
    def reflection_pad(source_dimension, target_dimension):
        dif = target_dimension - source_dimension
        padding_left = padding_right = padding_top = padding_bottom = dif // 2
        if dif % 2: padding_right = padding_bottom = (dif // 2) + 1
        return ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))

    def __init__(self, list_channels, encoder_stride, output_dimension, resnet_f=2):
        super(Decoder, self).__init__()

        hidden_channels = list_channels[1:]
        encoder_output_channel = list_channels[0]
        
        kernel_size = output_dimension//encoder_stride
        # print(kernel_size)
        self.feature = Linear(encoder_output_channel, encoder_output_channel * kernel_size * kernel_size)        
        self.unflatten = Unflatten(1, (encoder_output_channel, kernel_size, kernel_size))
        
        layers = OrderedDict()
        # layers['convT0'] = ConvTransposeBlock(encoder_output_channel, hidden_channels[0], kernel_size = kernel_size,
        #                                           padding=0, stride=1,
        #                                           activation = False, norm_type = constants.NO_NORM)
        
        n = 1
        for ci, co in zip(hidden_channels[:-2], hidden_channels[1:-1]):
            layers[f'resnet{n}'] = ResNetBlock(ci, f=resnet_f)
            layers[f'convT{n}'] = ConvTransposeBlock(ci, co)
            n+=1

        layers[f'resnet{n}'] = ResNetBlock(hidden_channels[-2], f=resnet_f)
        layers[f'reflect{n}'] = self.reflection_pad(encoder_stride * kernel_size, output_dimension)
        layers[f'convT{n}'] = ConvTransposeBlock(hidden_channels[-2], hidden_channels[-1], kernel_size=3,
                                                  padding=1, stride=1,
                                                  activation = False, norm_type = constants.NO_NORM)
        self.layers = Sequential(layers)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.unflatten(x)
        x = self.layers(x)
        return x



if __name__ == "__main__":
    channels = 1
    output_channels = 10
    kernel_size = 4
    padding = 1
    stride = 2
    input_dim = 108

    from torch import rand
    x = rand(5, channels, input_dim, input_dim)
    print(x.shape)

    c = ConvBlock(channels, output_channels, activation=ReLU(inplace=True),
                  norm_type = constants.NO_NORM, kernel_size=kernel_size, stride=stride, padding=padding)
    # rn = ResNetBlock(output_channels, f=2)
    # ct = ConvTransposeBlock(output_channels, channels, kernel_size=kernel_size, padding=padding, stride=stride)
    # # print(c, rn, ct)
    # x = c(x)
    # print(x.shape)
    # x = rn(x)
    # print(x.shape)
    # x = ct(x)
    # print(x.shape)

    en_channels = [channels, 32, 64, 128, 256, 512, 512]
    de_channels = en_channels[::-1]
    de_channels.insert(-1, 32)
    print(en_channels, de_channels)

    e = Encoder(en_channels, input_dimension = input_dim)
    d = Decoder(de_channels, 2**(len(en_channels) - 2),  input_dim)
    # print(d)

    x = e(x)
    print('Encoder output:', x.shape)
    x = d(x)
    print('DEcoder output:', x.shape)



