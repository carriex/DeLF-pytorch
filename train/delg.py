
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time
sys.path.append('../')
import random
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from train.layers import (
    CMul, 
    Flatten, 
    ConcatTable, 
    Identity, 
    Reshape, 
    SpatialAttention2d, 
    WeightedSum2d,
    ConvAutoEncoder,
    GeneralizedMeanPooling)


''' helper functions
'''

def __unfreeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = True
    
def __freeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = False

def __print_freeze_status__(model):
    '''print freeze stagus. only for debugging purpose.
    '''
    for i, module in enumerate(model.named_children()):
        for param in module[1].parameters():
            print('{}:{}'.format(module[0], str(param.requires_grad)))

def __load_weights_from__(module_dict, load_dict, modulenames):
    for modulename in modulenames:
        module = module_dict[modulename]
        print('loaded weights from module "{}" ...'.format(modulename))
        module.load_state_dict(load_dict[modulename])

def __deep_copy_module__(module, exclude=[]):
    modules = {}
    for name, m in module.named_children():
        if name not in exclude:
            modules[name] = copy.deepcopy(m)
            print('deep copied weights from layer "{}" ...'.format(name))
    return modules

def __cuda__(model):
    if torch.cuda.is_available():
        model.cuda()
    return model

'''global feature loss function'''
class ArcFaceCrossEntropyLoss(nn.Module):
    def __init__(self, m):
        super(ArcFaceCrossEntropyLoss, self).__init__()
        self.m = m
        self.s = torch.randn(1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        if torch.cuda.is_available():
            self.s.cuda()
            self.log_softmax.cuda()
    def forward(self, x, label):
        '''x: (num_of_class,) label: (1,)'''
        # create one-hot encoding
        pred = torch.zeros(x.shape) # 8 x 1000
        if torch.cuda.is_available():
            pred = pred.cuda()
        label = torch.unsqueeze(label, 1)
        pred.scatter_(1, label, 1 )
        result = torch.cos(torch.acos(x) + self.m) * pred + (1-pred) * x
        return torch.mean(-1 * self.log_softmax(result))



'''DELG'''
class DELG(nn.Module):

    def __init__(
            self,
            ncls=None,
            load_from=None,
            arch='resnet50',
            stage='train',
            target_layer='layer3',
            use_random_gamma_rescale=False):

        super(DELG, self).__init__()

        self.arch = arch
        self.stage = stage
        self.target_layer = target_layer
        self.load_from = load_from
        self.use_random_gamma_rescale = use_random_gamma_rescale

        self.module_list = nn.ModuleList()
        self.module_dict = {}
        self.end_points = {}
        self.global_module_list = nn.ModuleList()
        self.local_module_list = nn.ModuleList()
        self.img_dim = 16 # for generalized mean pooling : hard code for now

        in_c = self.__get_in_c__()
        use_pretrained_base = True
        # exclude from copying model
        exclude = ['avgpool', 'fc']

        if self.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print('[{}] loading {} pretrained ImageNet weights ... It may take few seconds...'
                  .format(self.stage, self.arch))
            module = models.__dict__[self.arch](pretrained=use_pretrained_base)
            module_state_dict = __deep_copy_module__(module, exclude=exclude)
            module = None

            # endpoint: base
            submodules = []
            submodules.append(module_state_dict['conv1'])
            submodules.append(module_state_dict['bn1'])
            submodules.append(module_state_dict['relu'])
            submodules.append(module_state_dict['maxpool'])
            submodules.append(module_state_dict['layer1'])
            submodules.append(module_state_dict['layer2'])
            submodules.append(module_state_dict['layer3'])
            self.__register_module__('base', submodules, self.global_module_list)
            # self.__register_module__('base', submodules, 'local')

            # ------------global feature---------------#
            # extract deep features (with relu) - conv5
            # generalized pooling with whitening
            self.__register_module__('layer4', module_state_dict['layer4'], self.global_module_list)
            # g = F * (gem)^(1/p) + b_{f} (p = 3, F = 2048)
            self.__register_module__('gem', GeneralizedMeanPooling(self.img_dim, 3, 1e-6), self.global_module_list)
            self.__register_module__('whitening', nn.Linear(2048, 2048), self.global_module_list)
            self.__register_module__('global_fc', nn.Linear(2048, ncls, bias=False), self.global_module_list)
            # ------------local feature---------------#
            # extract shallow features - conv4
            # autoencoder 1024 -> 128
            self.__register_module__('convae', ConvAutoEncoder(in_dim=1024, out_dim=128), self.local_module_list)
            # attention layer
            self.__register_module__('attn', SpatialAttention2d(in_c=1024, act_fn='relu'), self.local_module_list)
            self.__register_module__('attn_pool', WeightedSum2d(), self.local_module_list)

            submodules = []
            submodules.append(nn.Conv2d(1024, ncls, 1))
            submodules.append(Flatten())
            self.__register_module__('local_fc', submodules, self.local_module_list)


            # inference time
            if self.stage in ['inference']:
                load_dict = torch.load(self.load_from)
                self.__load_weights_from__(load_dict, modulenames=['base'], feature_type='base')
                self.__load_weights_from__(load_dict, modulenames=['layer4', 'gem', 'whitening'], feature_type='global')
                self.__load_weights_from__(load_dict, modulenames=['attn', 'attn_pool', 'convae'], feature_type='global')
                print('load model from "{}"'.format(load_from))

    def __register_module__(self, modulename, module, module_list):
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        # self.module_list.append(module)
        self.module_dict[modulename] = module
        module_list.append(module)

    def __get_in_c__(self):
        # adjust input channels according to arch.
        if self.arch in ['resnet18', 'resnet34']:
            in_c = 512
        elif self.arch in ['resnet50', 'resnet101', 'resnet152']:
            if self.stage in ['finetune']:
                in_c = 2048
            elif self.stage in ['keypoint', 'inference']:
                if self.target_layer in ['layer3']:
                    in_c = 1024
                elif self.target_layer in ['layer4']:
                    in_c = 2048
        return in_c

    def __forward_and_save__(self, x, modulename):
        module = self.module_dict[modulename]
        x = module(x)
        self.end_points[modulename] = x
        return x

    def __forward_and_save_feature__(self, x, model, name):
        x = model(x)
        self.end_points[name] = x.data
        return x

    def __gamma_rescale__(self, x, min_scale=0.3535, max_scale=1.0):
        '''max_scale > 1.0 may cause training failure.
        '''
        h, w = x.size(2), x.size(3)
        assert w == h, 'input must be square image.'
        gamma = random.uniform(min_scale, max_scale)
        new_h, new_w = int(h * gamma), int(w * gamma)
        x = F.upsample(x, size=(new_h, new_w), mode='bilinear')
        return x

    def get_endpoints(self):
        return self.end_points

    def get_feature_at(self, modulename):
        return copy.deepcopy(self.end_points[modulename].data.cpu())

    def write_to(self, state):
        if self.stage in ['finetune']:
            state['base'] = self.module_dict['base'].state_dict()
            state['layer4'] = self.module_dict['layer4'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        elif self.stage in ['keypoint']:
            state['base'] = self.module_dict['base'].state_dict()
            state['attn'] = self.module_dict['attn'].state_dict()
            state['pool'] = self.module_dict['pool'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        else:
            assert self.stage in ['inference']
            raise ValueError('inference does not support model saving!')

    def forward_for_serving(self, x):
        '''
        This function directly returns attention score and raw features
        without saving to endpoint dict.
        '''
        x = self.__forward_and_save__(x, 'base')
        if self.target_layer in ['layer4']:
            x = self.__forward_and_save__(x, 'layer4')
        ret_x = x
        if self.use_l2_normalized_feature:
            attn_x = F.normalize(x, p=2, dim=1)
        else:
            attn_x = x
        attn_score = self.__forward_and_save__(x, 'attn')
        ret_s = attn_score
        return ret_x.data.cpu(), ret_s.data.cpu()

    def forward(self, x):
        # x: (3, 512, 512)
        if self.stage not in ['inference']:
            # ------------backbone--------------#
            x = self.__forward_and_save__(x, 'base')
            # ------------global feature--------------#
            global_x = self.__forward_and_save__(x, 'layer4')
            global_x = self.__forward_and_save__(global_x, 'gem')
            global_x = self.__forward_and_save__(global_x, 'whitening')
            global_feature = F.normalize(global_x, p=2, dim=1)
            global_classification_output = self.__forward_and_save__(global_feature, 'global_fc')
            # ------------auto encoder---------------#
            local_feature = x
            decoded_x = self.__forward_and_save__(x, 'convae')
            local_feature_reconstructed = F.normalize(decoded_x, p=2, dim=1)
            # ------------local feature---------------#
            # attention layer
            attn_x = local_feature_reconstructed
            attn_score = self.__forward_and_save__(local_feature_reconstructed, 'attn')
            local_attn = self.__forward_and_save__([attn_x, attn_score], 'attn_pool')
            local_classification_output = self.__forward_and_save__(local_attn, 'local_fc')

        # inference
        else:
            x = self.__forward_and_save__(x, 'base')
            if self.target_layer in ['layer4']:
                x = self.__forward_and_save__(x, 'layer4')
            if self.use_l2_normalized_feature:
                attn_x = F.normalize(x, p=2, dim=1)
            else:
                attn_x = x
            attn_score = self.__forward_and_save__(x, 'attn')
            x = self.__forward_and_save__([attn_x, attn_score], 'pool')

        return global_classification_output, local_classification_output, \
               local_feature_reconstructed, local_feature


if __name__=="__main__":
    pass;









