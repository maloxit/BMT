import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vT
import numpy as np


####################################################################
# ------------------------- SPL LOSS -------------------------------
####################################################################
class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, input, reference):
        temp_a = torch.sum(F.normalize(input, p=2.0, dim=2, eps=1e-4) * F.normalize(reference, p=2.0, dim=2, eps=1e-4),
                           2, keepdim=True)
        temp_b = torch.sum(F.normalize(input, p=2.0, dim=3, eps=1e-4) * F.normalize(reference, p=2.0, dim=3, eps=1e-4),
                           3, keepdim=True)
        a = torch.sum(temp_a)
        b = torch.sum(temp_b)
        B, c, h, w = input.shape
        return -(a + b) / h


class SPLossWeighted(nn.Module):
    def __init__(self):
        super(SPLossWeighted, self).__init__()

    def forward(self, input, reference, weights):
        temp_a = torch.sum(
            F.normalize(input, p=2.0, dim=2, eps=1e-4) * F.normalize(reference, p=2.0, dim=2, eps=1e-4) * weights,
            2, keepdim=True)
        temp_b = torch.sum(
            F.normalize(input, p=2.0, dim=3, eps=1e-4) * F.normalize(reference, p=2.0, dim=3, eps=1e-4) * weights,
            3, keepdim=True)
        a = torch.sum(temp_a)
        b = torch.sum(temp_b)
        B, c, h, w = input.shape
        return -(a + b) / h


class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self._w_trace = SPLossWeighted()
        self._trace = SPLoss()

    def trace(self, input, reference, weights):
        if weights is None:
            return self._trace(input, reference)
        else:
            return self._w_trace(input, reference, weights)

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def forward(self, input, reference, weights=None):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        if weights is None:
            trace_v = self.trace(input_v, ref_v, weights)
            trace_h = self.trace(input_h, ref_h, weights)
        else:
            trace_v = self.trace(input_v, ref_v, weights[:, :, :, :-1])
            trace_h = self.trace(input_h, ref_h, weights[:, :, :-1, :])
        return trace_v + trace_h


class CPLoss(nn.Module):
    def __init__(self, rgb=True, yuv=True, yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self._w_trace = SPLossWeighted()
        self._trace = SPLoss()

    def trace(self, input, reference, weights):
        if weights is None:
            return self._trace(input, reference)
        else:
            return self._w_trace(input, reference, weights)

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.cat([0.299 * input[:, None, 0, :, :] +
                          0.587 * input[:, None, 1, :, :] +
                          0.114 * input[:, None, 2, :, :],
                          0.493 * (input[:, None, 2, :, :] - (
                                  0.299 * input[:, None, 0, :, :] +
                                  0.587 * input[:, None, 1, :, :] +
                                  0.114 * input[:, None, 2, :, :])),
                          0.877 * (input[:, None, 0, :, :] - (
                                  0.299 * input[:, None, 0, :, :] +
                                  0.587 * input[:, None, 1, :, :] +
                                  0.114 * input[:, None, 2, :, :]))], dim=1)

    def forward(self, input, reference, weights=None):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference, weights)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv, weights)
        if self.yuvgrad:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)
            if weights is None:
                total_loss += self.trace(input_v, ref_v, weights)
                total_loss += self.trace(input_h, ref_h, weights)
            else:
                total_loss += self.trace(input_v, ref_v, weights[:, :, :, :-1])
                total_loss += self.trace(input_h, ref_h, weights[:, :, :-1, :])

        return total_loss


class WeightMaskGenerator(nn.Module):
    def __init__(self):
        super(WeightMaskGenerator, self).__init__()
        self.big_blur = vT.GaussianBlur(51, 15.)
        self.small_blur = vT.GaussianBlur(21, 2.)

    def batch_max_normalization(self, input):
        tmp = input.view(input.size(0), -1)
        tmp = tmp / tmp.max(1, keepdim=True)[0]
        return tmp.view(input.shape)

    def batch_mean_normalization(self, input):
        tmp = input.view(input.size(0), -1)
        tmp = tmp / tmp.mean(1, keepdim=True)
        return tmp.view(input.shape)

    def forward(self, mask_parse, area_weights, eye_shadows_weight):
        background_mask = mask_parse[:, 0, :, :].unsqueeze(1)
        eyes_mask_l = mask_parse[:, 4, :, :].unsqueeze(1)
        w_mask_l = self.big_blur(eyes_mask_l)
        w_mask_l = w_mask_l * (1 - eyes_mask_l) * (1 - background_mask)
        w_mask_l = self.batch_max_normalization(w_mask_l)

        eyes_mask_r = mask_parse[:, 5, :, :].unsqueeze(1)
        w_mask_r = self.big_blur(eyes_mask_r)
        w_mask_r = w_mask_r * (1 - eyes_mask_r) * (1 - background_mask)
        w_mask_r = self.batch_max_normalization(w_mask_r)

        mask = torch.zeros_like(mask_parse[:, 0:1, :, :])
        for i in range(len(area_weights)):
            mask = mask + (mask_parse[:, i, :, :] * area_weights[i]).unsqueeze(1)
        mask = mask + w_mask_l * eye_shadows_weight + w_mask_r * eye_shadows_weight
        mask = self.small_blur(mask)
        mask = self.batch_mean_normalization(mask)
        return mask


####################################################################
# ------------------------- GANLoss -------------------------------
####################################################################

class GANLoss(nn.Module):
    def __init__(self, mode="lsgan", reduction='mean'):
        super(GANLoss, self).__init__()
        self.loss = None
        if mode == "lsgan":
            self.loss = nn.MSELoss(reduction=reduction)
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            raise NotImplementedError(f'GANLoss {mode} not recognized, we support lsgan and vanilla.')

    def forward(self, predict, target):
        target = target.type_as(predict)
        target = torch.ones_like(predict) * target
        loss = self.loss(predict, target)
        return loss


####################################################################
# ------------------------- D_Loss -------------------------------
####################################################################

class SAATDLoss(nn.Module):
    def __init__(self, opts, dis_non_makeup, dis_makeup):
        super(SAATDLoss, self).__init__()
        self.opts = opts
        self.dis_non_makeup = dis_non_makeup
        self.dis_makeup = dis_makeup

        self.dis_loss = GANLoss(opts.gan_mode)
        self.false = torch.BoolTensor([False])
        self.true = torch.BoolTensor([True])

    def forward(self, non_makeup, makeup, z_transfer, z_removal):
        non_makeup_real = self.dis_non_makeup(non_makeup)
        non_makeup_fake = self.dis_non_makeup(z_removal)
        makeup_real = self.dis_makeup(makeup)
        makeup_fake = self.dis_makeup(z_transfer)
        loss_D_non_makeup = self.dis_loss(non_makeup_fake, self.false) + self.dis_loss(non_makeup_real, self.true)
        loss_D_makeup = self.dis_loss(makeup_fake, self.false) + self.dis_loss(makeup_real, self.true)
        loss_D = (loss_D_makeup + loss_D_non_makeup) * 0.5
        return loss_D


class SAATGLoss(nn.Module):
    def __init__(self, opts, generator, dis_non_makeup, dis_makeup):
        super(SAATGLoss, self).__init__()
        self.opts = opts

        self.gen = generator
        self.dis_non_makeup = dis_non_makeup
        self.dis_makeup = dis_makeup

        self.adv_loss = GANLoss(opts.gan_mode)
        self.criterionL1 = nn.L1Loss()
        self.weight_mask_gen = WeightMaskGenerator()
        self.GPL = GPLoss()
        self.CPL = CPLoss(rgb=True, yuv=True, yuvgrad=True)

        self.CP_weight = opts.CP_weight
        self.GP_weight = opts.GP_weight
        self.rec_weight = opts.rec_weight
        self.cycle_weight = opts.cycle_weight
        self.semantic_weight = opts.semantic_weight
        self.adv_weight = opts.adv_weight

        # mask attribute: 0:background 1:face 2:left-eyebrow 3:right-eyebrow 4:left-eye 5: right-eye 6: nose
        # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
        self.area_weights = [0.05, 0.75, 1., 1., 1., 1., 1., 2., 1, 2., 0.2, 1., 1., 0.5]
        self.eye_shadows_weight = 2.

        self.false = torch.BoolTensor([False])
        self.true = torch.BoolTensor([True])

    def forward(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse,
                loss_hyperparams = None, generator_output: tuple = None):
        if generator_output is None:
            z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = \
                self.gen(non_makeup, makeup, non_makeup_parse, makeup_parse)
        else:
            z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = \
                generator_output

        # Ladv for generator
        loss_G_GAN_non_makeup = self.adv_loss(self.dis_non_makeup(z_removal), self.true)
        loss_G_GAN_makeup = self.adv_loss(self.dis_makeup(z_transfer), self.true)
        loss_G_GAN = (loss_G_GAN_non_makeup + loss_G_GAN_makeup) * 0.5

        # rec loss
        loss_G_rec_non_makeup = self.criterionL1(non_makeup, z_rec_non_makeup)
        loss_G_rec_makeup = self.criterionL1(makeup, z_rec_makeup)
        loss_G_rec = (loss_G_rec_non_makeup + loss_G_rec_makeup) * 0.5

        # cycle loss
        loss_G_cycle_non_makeup = self.criterionL1(non_makeup, z_cycle_non_makeup)
        loss_G_cycle_makeup = self.criterionL1(makeup, z_cycle_makeup)
        loss_G_cycle = (loss_G_cycle_non_makeup + loss_G_cycle_makeup) * 0.5

        # semantic loss
        non_makeup_parse_down = F.interpolate(non_makeup_parse,
                                              size=(self.opts.crop_size // 4, self.opts.crop_size // 4))
        n, c, h, w = non_makeup_parse_down.shape
        non_makeup_parse_down_warp = torch.bmm(non_makeup_parse_down.reshape(n, c, h * w), mapY)  # n*HW*1
        non_makeup_parse_down_warp = non_makeup_parse_down_warp.reshape(n, c, h, w)

        makeup_parse_down = F.interpolate(makeup_parse, size=(self.opts.crop_size // 4, self.opts.crop_size // 4))
        n, c, h, w = makeup_parse_down.shape
        makeup_parse_down_warp = torch.bmm(makeup_parse_down.reshape(n, c, h * w), mapX)  # n*HW*1
        makeup_parse_down_warp = makeup_parse_down_warp.reshape(n, c, h, w)

        loss_G_semantic_non_makeup = self.criterionL1(non_makeup_parse_down, makeup_parse_down_warp)
        loss_G_semantic_makeup = self.criterionL1(makeup_parse_down, non_makeup_parse_down_warp)
        loss_G_semantic = (loss_G_semantic_makeup + loss_G_semantic_non_makeup) * 0.5

        # makeup loss
        if loss_hyperparams is None:
            non_makeup_weights = self.weight_mask_gen(non_makeup_parse, self.area_weights, self.eye_shadows_weight)
            makeup_weights = self.weight_mask_gen(makeup_parse, self.area_weights, self.eye_shadows_weight)
        else:
            non_makeup_weights = self.weight_mask_gen(non_makeup_parse, loss_hyperparams['area_weights'], loss_hyperparams['eye_shadows_weight'])
            makeup_weights = self.weight_mask_gen(makeup_parse, loss_hyperparams['area_weights'], loss_hyperparams['eye_shadows_weight'])

        loss_G_CP = self.CPL(z_transfer, transfer, non_makeup_weights) + self.CPL(z_removal, removal, makeup_weights)
        loss_G_GP = self.GPL(z_transfer, non_makeup, non_makeup_weights) + self.GPL(z_removal, makeup, makeup_weights)

        if loss_hyperparams is None:
            loss_G_GAN *= self.adv_weight
            loss_G_rec *= self.rec_weight
            loss_G_cycle *= self.cycle_weight
            loss_G_semantic *= self.semantic_weight
            loss_G_SPL = loss_G_CP * self.CP_weight + loss_G_GP * self.GP_weight
        else:
            loss_G_GAN *= loss_hyperparams['adv_weight']
            loss_G_rec *= loss_hyperparams['rec_weight']
            loss_G_cycle *= loss_hyperparams['cycle_weight']
            loss_G_semantic *= loss_hyperparams['semantic_weight']
            loss_G_SPL = loss_G_CP * loss_hyperparams['CP_weight'] + loss_G_GP * loss_hyperparams['GP_weight']


        loss_G = loss_G_GAN + loss_G_rec + loss_G_cycle + loss_G_semantic + loss_G_SPL

        loss_distr = ((loss_G_GAN / loss_G).item(), (loss_G_rec / loss_G).item(), (loss_G_cycle / loss_G).item(), (loss_G_semantic / loss_G).item(), (loss_G_SPL / loss_G).item())

        return loss_G, loss_distr, z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, \
               mapX, mapY


if __name__ == '__main__':
    a = torch.tensor(
        np.array([[[[1, 1, 3], [4, 5, 6]], [[1, 1, 3], [4, 5, 6]], [[1, 1, 3], [4, 5, 6]]]]).astype(np.float32))
    b = torch.tensor(
        np.array([[[[7, 8, 5], [10, 3, 12]], [[7, 8, 5], [10, 3, 12]], [[7, 8, 5], [10, 3, 12]]]]).astype(np.float32))

    print(a.shape)
    SPL = SPLoss()
    GP = GPLoss()
    CP = CPLoss()
    c = GP(a, b)
    d = CP(a, b)
    e = SPL(a, b)
    print(c)
    print(d)
    print(e)
    # pad_op_x1 = ops.Pad(((0, 0), (0, 0), (0, 0), (0, 1)))
    # f=pad_op_x1(a)
    # g=a[:,:,:,:-1]
    # print(f.shape)
    # print(f)
    # print(g.shape)
    # print(g)
