#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PyTorch Implementation of training DeLF feature.
Solver for step 1 (finetune local descriptor)
nashory, 2018.04
'''
import os, sys, time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from delg import ArcFaceMargin

from utils import Bar, Logger, AverageMeter, compute_precision_top_k, mkdir_p

'''helper functions.
'''


def __cuda__(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def __is_cuda__():
    return torch.cuda.is_available()


def __to_var__(x, volatile=False):
    return Variable(x, volatile=volatile)


def __to_tensor__(x):
    return x.data


def __unfreeze_weights__(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = True


def __freeze_weights__(module_list):
    for module in module_list:
        for param in module.parameters():
            param.requires_grad = False


class Solver(object):
    def __init__(self, config, model):
        self.state = {k: v for k, v in config._get_kwargs()}
        self.config = config
        self.epoch = 0  # global epoch.
        self.best_acc = 0  # global best accuracy.
        self.prefix = os.path.join('repo', config.expr)
        num_epoches = config.finetune_epoch

        # ship model to cuda
        self.model = __cuda__(model)

        # ------------DELG loss---------------#
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.AFMargin = ArcFaceMargin(m=0.1)
        self.mse_loss = nn.MSELoss()

        if torch.cuda.is_available():
            self.cross_entropy_loss.cuda()
            self.AFMargin.cuda()
            self.mse_loss.cuda()

        # ------------DELG optimizer---------------#
        self.global_feature_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.global_module_list.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay) #remove momentum

        self.local_feature_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.local_module_list.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay) #remove momentum

        # decay learning rate by a factor of 0.5 every 10 epochs
        self.global_feature_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.global_feature_optimizer,
            lr_lambda=(lambda epoch: (1 - float(epoch) / float(num_epoches))))

        self.local_feature_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.local_feature_optimizer,
            lr_lambda=(lambda epoch: (1 - float(epoch) / float(num_epoches))))

        # create directory to save result if not exist.
        self.ckpt_path = os.path.join(self.prefix, config.stage, 'ckpt')
        self.log_path = os.path.join(self.prefix, config.stage, 'log')
        self.image_path = os.path.join(self.prefix, config.stage, 'image')
        mkdir_p(self.ckpt_path)
        mkdir_p(self.log_path)
        mkdir_p(self.image_path)

        # set logger.
        self.logger = {}
        self.title = 'DeLF-{}'.format(config.stage.upper())
        self.logger['train'] = Logger(os.path.join(self.prefix, config.stage, 'log/train.log'))
        self.logger['val'] = Logger(os.path.join(self.prefix, config.stage, 'log/val.log'))

        self.logger['train'].set_names(
            ['epoch', 'global_lr', 'local_lr', 'global_losses', 'local_losses'])
        self.logger['val'].set_names(
            ['epoch', 'global_lr', 'local_lr', 'global_losses', 'local_losses'])

    def __exit__(self):
        self.train_logger.close()
        self.val_logger.close()

    def __adjust_pixel_range__(self,
                               x,
                               range_from=[0, 1],
                               range_to=[-1, 1]):
        '''
        adjust pixel range from <range_from> to <range_to>.
        '''
        if not range_from == range_to:
            scale = float(range_to[1] - range_to[0]) / float(range_from[1] - range_from[0])
            bias = range_to[0] - range_from[0] * scale
            x = x.mul(scale).add(bias)
            return x

    def __save_checkpoint__(self, state, ckpt='ckpt', filename='checkpoint.pth.tar'):
        filepath = os.path.join(ckpt, filename)
        torch.save(state, filepath)

    def __solve__(self, mode, epoch, dataloader):
        '''solve
        mode: train / val
        '''
        batch_timer = AverageMeter()
        data_timer = AverageMeter()
        global_losses = AverageMeter()
        local_losses = AverageMeter()
        local_prec_top1 = AverageMeter()
        local_prec_top3 = AverageMeter()
        local_prec_top5 = AverageMeter()
        global_prec_top1 = AverageMeter()
        global_prec_top3 = AverageMeter()
        global_prec_top5 = AverageMeter()

        since = time.time()
        bar = Bar('[{}]{}'.format(mode.upper(), self.title), max=len(dataloader))


        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # measure data loading time
            data_timer.update(time.time() - since)

            # wrap inputs in variable
            if mode in ['train']:
                if __is_cuda__():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs = __to_var__(inputs)
                labels = __to_var__(labels)
            elif mode in ['val']:
                if __is_cuda__():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs = __to_var__(inputs, volatile=True)
                labels = __to_var__(labels, volatile=False)

            # forward
            global_classification_output, local_classification_output, local_feature_reconstructed, local_feature = self.model(
               inputs)
            # global_classification_output = self.model(inputs)

            # backward + optimize
            if mode in ['train']:
                # ------------global feature---------------#
                self.global_feature_optimizer.zero_grad()
                # __unfreeze_weights__(self.model.global_module_list)
                # l_g = self.cross_entropy_loss(self.AFMargin(global_classification_output, labels), labels)
                l_g = self.cross_entropy_loss(global_classification_output, labels)
                l_g.backward(retain_graph=True) # as we will call again
                self.global_feature_optimizer.step()
                # ------------local feature---------------#
                self.local_feature_optimizer.zero_grad()
                __freeze_weights__(self.model.global_module_list)
                H_S, W_S, C_S = local_feature.shape[1:]
                l_r = (1 / H_S * W_S * C_S) * self.mse_loss(local_feature_reconstructed, local_feature)
                l_a = self.cross_entropy_loss(local_classification_output, labels)
                #l_l = 10 * l_r + 1 * l_a
                l_l = l_a
                l_l.backward()
                self.local_feature_optimizer.step()
                __unfreeze_weights__(self.model.global_module_list)

            local_prec_1, local_prec_3, local_prec_5 = compute_precision_top_k(
                __to_tensor__(local_classification_output),
                __to_tensor__(labels),
                top_k=(1, 3, 5))

            global_prec_1, global_prec_3, global_prec_5 = compute_precision_top_k(
                __to_tensor__(global_classification_output),
                __to_tensor__(labels),
                top_k=(1, 3, 5))

            batch_size = inputs.size(0)

            if mode in ['train']:
                global_losses.update(l_g.item(), batch_size)
                local_losses.update(l_l.item(), batch_size)

            local_prec_top1.update(local_prec_1, batch_size)
            local_prec_top3.update(local_prec_3, batch_size)
            local_prec_top5.update(local_prec_5, batch_size)
            global_prec_top1.update(global_prec_1, batch_size)
            global_prec_top3.update(global_prec_3, batch_size)
            global_prec_top5.update(global_prec_5, batch_size)

            # measure elapsed time
            batch_timer.update(time.time() - since)
            since = time.time()

            # progress
            log_msg = ('\n[{mode}][epoch:{epoch}][iter:({batch}/{size})]' +
                       '[global_lr:{global_lr}] [local_lr:{local_lr}]' +
                        'global loss: {global_loss:.4f} | local loss: {local_loss:.4f} | ' +
                        '[g_top_1:{g_top1}] [g_top_3:{g_top3}] [g_top_5:{g_top5}]' +
                        '[l_top_1:{l_top1}] [l_top_3:{l_top3}] [l_top_5:{l_top5}]' +
                       'eta: ' +
                       '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
                .format(
                mode=mode,
                epoch=self.epoch + 1,
                batch=batch_idx + 1,
                size=len(dataloader),
                global_lr=self.global_feature_lr_scheduler.get_lr()[0],
                local_lr=self.local_feature_lr_scheduler.get_lr()[0],
                global_loss=global_losses.avg,
                local_loss=local_losses.avg,
                g_top1 = global_prec_top1.avg,
                g_top3 = global_prec_top3.avg,
                g_top5= global_prec_top5.avg,
                l_top1 = local_prec_top1.avg,
                l_top3 = local_prec_top3.avg,
                l_top5 = local_prec_top5.avg,
                dt=data_timer.val,
                bt=batch_timer.val,
                tt=bar.elapsed_td)
            print(log_msg)
            bar.next()
        bar.finish()

        #write to logger
        self.logger[mode].append([self.epoch + 1,
                                  self.global_feature_lr_scheduler.get_lr()[0],
                                  self.local_feature_lr_scheduler.get_lr()[0],
                                  global_losses.avg,
                                  local_losses.avg])

        # save model
        if mode == 'train':
            state = {
                'epoch': self.epoch,
            }
            self.model.write_to(state)
            filename = 'model.pth.tar'
            self.__save_checkpoint__(state, ckpt=self.ckpt_path, filename=filename)

            self.global_feature_lr_scheduler.step()
            self.local_feature_lr_scheduler.step()

    def train(self, mode, epoch, train_loader, val_loader):
        self.epoch = epoch
        if mode in ['train']:
            self.model.train()
            dataloader = train_loader
        else:
            assert mode == 'val'
            self.model.eval()
            dataloader = val_loader
        self.__solve__(mode, epoch, dataloader)
