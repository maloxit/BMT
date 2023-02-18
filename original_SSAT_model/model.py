#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhaoyang Sun

import original_SSAT_model.networks as networks
from original_SSAT_model.networks import init_net
import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeupGAN(nn.Module):
    def __init__(self, opts):
        super(MakeupGAN, self).__init__()
        self.opts = opts

        # parameters
        self.lr = opts.lr
        self.batch_size = opts.batch_size

        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim

        # encoders
        self.enc_content = init_net(networks.E_content(opts.input_dim), self.gpu, init_type='normal', gain=0.02)
        self.enc_makeup = init_net(networks.E_makeup(opts.input_dim), self.gpu, init_type='normal', gain=0.02)
        self.enc_semantic = init_net(networks.E_semantic(opts.semantic_dim), self.gpu, init_type='normal', gain=0.02)
        self.transformer = init_net(networks.Transformer(), self.gpu, init_type='normal', gain=0.02)
        # generator
        self.gen = init_net(networks.Decoder(opts.output_dim), self.gpu, init_type='normal', gain=0.02)


    def get_transfers(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)

        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer(z_non_makeup_c,
                                                                                      z_makeup_c,
                                                                                      z_non_makeup_s,
                                                                                      z_makeup_s,
                                                                                      z_non_makeup_a,
                                                                                      z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)
        return z_transfer, z_removal

    def forward(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        # first transfer and removal
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)

        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer(z_non_makeup_c,
                                                                                      z_makeup_c,
                                                                                      z_non_makeup_s,
                                                                                      z_makeup_s,
                                                                                      z_non_makeup_a,
                                                                                      z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)

        # rec
        z_rec_non_makeup = self.gen(z_non_makeup_c, z_non_makeup_a)
        z_rec_makeup = self.gen(z_makeup_c, z_makeup_a)

        # second transfer and removal
        z_transfer_c = self.enc_content(z_transfer)
        z_transfer_a = self.enc_makeup(z_transfer)

        z_removal_c = self.enc_content(z_removal)
        z_removal_a = self.enc_makeup(z_removal)
        # warp makeup style
        mapX2, mapY2, z_transfer_a_warp, z_removal_a_warp = self.transformer(z_transfer_c, z_removal_c, z_non_makeup_s,
                                                                             z_makeup_s, z_transfer_a, z_removal_a)

        # makeup transfer and removal
        z_cycle_non_makeup = self.gen(z_transfer_c, z_removal_a_warp)
        z_cycle_makeup = self.gen(z_removal_c, z_transfer_a_warp)
        return z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        # weight
        self.enc_content.load_state_dict(checkpoint['enc_c'])
        self.enc_makeup.load_state_dict(checkpoint['enc_a'])
        self.enc_semantic.load_state_dict(checkpoint['enc_s'])
        self.transformer.load_state_dict(checkpoint['enc_trans'])
        self.gen.load_state_dict(checkpoint['gen'])
        return checkpoint['ep'], checkpoint['total_it']
