import networks
from networks import init_net
import torch
import torch.nn as nn


class SSAT_D_non_makeup(nn.Module):
    def __init__(self, opts):
        super(SSAT_D_non_makeup, self).__init__()
        self.dis_non_makeup = None
        if opts.dis_scale > 1:
            self.dis_non_makeup = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale)
        else:
            self.dis_non_makeup = networks.Dis(opts.input_dim)

    def forward(self, x):
        return self.dis_non_makeup(x)


class SSAT_D_makeup(nn.Module):
    def __init__(self, opts):
        super(SSAT_D_makeup, self).__init__()
        self.dis_makeup = None
        if opts.dis_scale > 1:
            self.dis_makeup = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale)
        else:
            self.dis_makeup = networks.Dis(opts.input_dim)

    def forward(self, x):
        return self.dis_makeup(x)


class SSAT_G(nn.Module):
    def __init__(self, opts):
        super(SSAT_G, self).__init__()
        self.opts = opts
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim
        # encoders
        self.enc_content = networks.E_content(opts.input_dim)
        self.enc_makeup = networks.E_makeup(opts.input_dim)
        self.enc_semantic = networks.E_semantic(opts.semantic_dim)
        # FF and SSCFT
        self.transformer = networks.Transformer()
        # generator
        self.gen = networks.Decoder(opts.output_dim)

    def output_fake_images(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
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

    def forward(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse):
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

    def output(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
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


####################################################################
# -------------------------- get_model --------------------------
####################################################################



def get_generator(opts, device):
    """Return generator by args."""
    net = SSAT_G(opts)
    init_net(net, device, opts.init_type, opts.init_gain)
    return net


def get_dis_non_makeup(opts, device):
    """Return discriminator by args."""
    net = SSAT_D_non_makeup(opts)
    init_net(net, device, opts.init_type, opts.init_gain)
    return net


def get_dis_makeup(opts, device):
    """Return discriminator by args."""
    net = SSAT_D_makeup(opts)
    init_net(net, device, opts.init_type, opts.init_gain)
    return net
