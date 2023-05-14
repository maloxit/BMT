import matplotlib.pyplot as plt
import torch
import torch.utils.data as torchdt
import argparse
import numpy as np
from model.loss import WeightMaskGenerator
from model.dataset import MakeupDataset
import torch.nn.functional as F
from tqdm import tqdm

def get_torch_device(opts):
    if opts.platform == 'GPU' and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(opts.device_id))
    else:
        return torch.device('cpu')
    
test_opts = argparse.Namespace(**{
    'warp_dir': 'datasets/test/images/warp_tmp',
    'warp_alt_dir': 'datasets/test/images/warp',
    'warp_storage_dir': 'content',
    'subset_config_files': 
    [
        'datasets/test_makeup.json',
        'datasets/test_non_makeup.json'
    ],
    'input_dim': 3,
    'output_dim': 3,
    'semantic_dim': 18,
    'batch_size': 1,
    'resize_size': 286,
    'crop_size': 256,
    'flip': True,
    'nThreads': 0,

    # platform related
    'platform': 'GPU',
    'device_id': 0,
    'device_num': 1,

    # ouptput related
    'name': 'BMT_ND_WM',
    'outputs_dir': 'content',
    'print_iter': 1,
    'save_imgs': True,
    'save_checkpoint_epochs': 5,

    # weight
    'gan_mode': 'lsgan',
    'rec_weight': 1,
    'CP_weight': 0.05,
    'GP_weight': 0.025,
    'cycle_weight': 1,
    'adv_weight': 1,
    'semantic_weight': 1,

    # training related
    'init_type': 'normal',
    'init_gain': 0.02,
    'beta1': 0.5,
    'beta2': 0.999,

    'dis_scale': 3,
    'dis_norm': 'None',
    'dis_spectral_norm': True,
    'lr_policy': 'lambda',
    'max_epoch': 1000,
    'n_epochs': 1000,
    'n_epochs_decay': 500,

    'resume': None,
    'num_residule_block': 4,
    'lr': 0.0002
})

device = get_torch_device(test_opts)

dataset = MakeupDataset(test_opts, device, ['datasets/paper_non_makeup.json', 'datasets/paper_makeup.json'], transform=False, need_pgt=True, all_comb=True, add_original_parsing=True)
data_loader = torchdt.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=test_opts.nThreads)

weightGen = WeightMaskGenerator()


def batch_max_normalization(input):
    tmp = input.view(input.size(0), -1)
    tmp = tmp / tmp.max(1, keepdim=True)[0]
    return tmp.view(input.shape)

area_weights = [0.05, 0.75, 1., 1., 1.5, 1.5, 1., 2., 1, 2., 0.2, 1., 1., 0.5]
eye_shadows_weight = 2.

def show_mask(data2d):
    fig, ax = plt.subplots()
    #im = ax.imshow(data2d, cmap='plasma', vmin=-0.03, vmax=0.01)
    im = ax.imshow(data2d, cmap='plasma')

    fig.colorbar(im, ax=ax)


def forward(input, reference, weights):
    B, c, h, w = input.shape
    loss = 0.0
    max_loss = 0.0
    inputT = input.transpose(2,3).contiguous()
    referenceT = reference.transpose(2,3).contiguous()
    weightsT = weights.transpose(2,3).contiguous()
    for k in range(0,6):
        div = int(2**k)

        i_h = input.view(B,c,-1,h//div)
        r_h = reference.view(B,c,-1,h//div)
        ir_hnorm = i_h.norm(p=2.0, dim=3, keepdim=True) * r_h.norm(p=2.0, dim=3, keepdim=True)
        ir_hnorm[ir_hnorm<1e-8] = 1.
        w_hsum = torch.sum(weights.view(B,1,-1,h//div), 3, keepdim=True) / h
        ir_hcos = i_h * r_h / ir_hnorm
        spl_h = (1-torch.sum(ir_hcos, 3, keepdim=True)) * w_hsum
        
        i_v = inputT.view(B,c,-1,h//div)
        r_v = referenceT.view(B,c,-1,h//div)
        ir_vnorm = i_v.norm(p=2.0, dim=3, keepdim=True) * r_v.norm(p=2.0, dim=3, keepdim=True)
        ir_vnorm[ir_vnorm<1e-8] = 1.
        w_vsum = torch.sum(weightsT.view(B,1,-1,h//div), 3, keepdim=True) / h
        ir_vcos = i_v * r_v / ir_vnorm
        spl_v = (1-torch.sum(ir_vcos, 3, keepdim=True)) * w_vsum

        loss += (spl_h.sum() + spl_v.sum()) / (c * h * 2) / div
        max_loss += 1 / div
    return loss / max_loss


def get_loss_map(input, reference, weights, depth=6):
    B, c, h, w = input.shape
    loss_maps = []
    loss_map_sum = torch.zeros_like(input)
    sum = 0.0
    max_sum = 0.0
    inputT = input.transpose(2,3).contiguous()
    referenceT = reference.transpose(2,3).contiguous()
    weightsT = weights.transpose(2,3).contiguous()
    for k in range(0,depth):
        loss_map = torch.zeros_like(input)
        div = int(2**k)

        i_h = input.view(B,c,-1,h//div)
        r_h = reference.view(B,c,-1,h//div)
        ir_hnorm = i_h.norm(p=2.0, dim=3, keepdim=True) * r_h.norm(p=2.0, dim=3, keepdim=True)
        ir_hnorm = torch.where(ir_hnorm<1e-8, 1.0, ir_hnorm)
        w_hsum = torch.sum(weights.view(B,1,-1,h//div), 3, keepdim=True) / h
        ir_hcos = i_h * r_h / ir_hnorm
        spl_h = (1/(h//div)-ir_hcos) * w_hsum
        
        i_v = inputT.view(B,c,-1,h//div)
        r_v = referenceT.view(B,c,-1,h//div)
        ir_vnorm = i_v.norm(p=2.0, dim=3, keepdim=True) * r_v.norm(p=2.0, dim=3, keepdim=True)
        ir_vnorm = torch.where(ir_vnorm<1e-8, 1.0, ir_vnorm)
        w_vsum = torch.sum(weightsT.view(B,1,-1,h//div), 3, keepdim=True) / h
        ir_vcos = i_v * r_v / ir_vnorm
        spl_v = (1/(h//div)-ir_vcos) * w_vsum

        #show_mask((spl_h.view(B,c,h,h).transpose(2,3) + spl_v.view(B,c,h,h))[0].sum(0).cpu())


        spl_h = spl_h.sum(3, keepdim=True)
        spl_v = spl_v.sum(3, keepdim=True)


        h_grid = spl_h.view(B,c,-1,div)
        #h_grid = torch.sum(h_grid.view(B,c,-1,h//div), 3, keepdim=True).view(B,c,div,div) / h
        v_grid = spl_v.view(B,c,-1,div).transpose(2,3)
        #v_grid = torch.sum(v_grid.view(B,c,-1,h//div), 3, keepdim=True).view(B,c,div,div) / h

        for i in range(0, div):
            for j in range(0, div):
                x_min = h//div * i
                x_max = h//div * (i+1)
                y_min = h//div * j
                y_max = h//div * (j+1)
                loss_map[:,:,x_min:x_max,y_min:y_max] = (h_grid[:,:,x_min:x_max,j:j+1].expand(B,c,h//div,h//div) + v_grid[:,:,i:i+1,y_min:y_max].expand(B,c,h//div,h//div)) / (2 * (h) * (h//div))
                #loss_map[:,:,x_min:x_max,y_min:y_max] = (h_grid[:,:,i:i+1,y_min:y_max].sum(3, keepdim=True) + v_grid[:,:,x_min:x_max,j:j+1].sum(2, keepdim=True)) / (2 * (h) * (h//div) * (h//div))
        loss_map_sum += loss_map / div
        loss_maps.append(loss_map)
        sum += (spl_h.sum() + spl_v.sum()) / (c * h * 2) / div
        max_sum += 1 / div
        #show_mask(loss_map[0].sum(0).cpu())
        #plt.show()
    
    return sum / max_sum, loss_map_sum / max_sum, loss_maps


with torch.no_grad():
    for i, data in enumerate(tqdm(data_loader)):
        non_makeup = data['non_makeup'].to(device)
        makeup = data['makeup'].to(device)
        transfer_g = data['transfer'].to(device)
        removal_g = data['removal'].to(device)
        non_makeup_parse = data['non_makeup_parse'].to(device)
        makeup_parse = data['makeup_parse'].to(device)

        non_makeup = ((non_makeup + 1)/2)
        makeup = ((makeup + 1)/2)
        transfer_g = ((transfer_g + 1)/2)
        removal_g = ((removal_g + 1)/2)

        fig, ax = plt.subplots(1, 2, gridspec_kw={'wspace':0.0,'hspace':0.0})
        for axis in ax:
            axis.set_axis_off()

        weights1 = weightGen.forward(non_makeup_parse, area_weights, eye_shadows_weight)
        weights2 = weightGen.forward(makeup_parse, area_weights, eye_shadows_weight)

        im = ax[0].imshow(weights1[0,0].cpu(), cmap='plasma', vmin=0, vmax=5)
        im = ax[1].imshow(weights2[0,0].cpu(), cmap='plasma', vmin=0, vmax=5)
        
        fig.colorbar(im, ax=ax)

        fig, ax = plt.subplots(2, 4, gridspec_kw={'wspace':0.0,'hspace':0.1})
        for axis_row in ax:
            for axis in axis_row:
                axis.set_axis_off()
        im = ax[0,0].imshow(non_makeup[0].cpu().permute(1,2,0))
        im = ax[1,0].imshow(transfer_g[0].cpu().permute(1,2,0))
        im = ax[0,1].imshow(weights1[0,0].cpu(), cmap='plasma', vmin=0, vmax=5)
        fig.colorbar(im, ax=ax[0,:2])

        im = ax[0,2].imshow(((non_makeup - transfer_g)/256/256).abs().sum(1)[0].cpu(), cmap='plasma', vmin=0, vmax=1e-4)
        im = ax[0,3].imshow(((non_makeup - transfer_g)*weights1/256/256).abs().sum(1)[0].cpu(), cmap='plasma', vmin=0, vmax=1e-4)
        
        fig.colorbar(im, ax=ax[0,2:])


        loss, loss_map_1, _ = get_loss_map(non_makeup, transfer_g, torch.ones_like(weights1), 1)
        print(loss)
        print(loss_map_1.sum() / 3)
        loss, loss_map_2, _ = get_loss_map(non_makeup, transfer_g, weights1, 1)
        print(loss)
        print(loss_map_2.sum() / 3)


        im = ax[1,2].imshow(loss_map_1[0].sum(0).cpu(), cmap='plasma', vmin=0, vmax=3e-6)
        im = ax[1,3].imshow(loss_map_2[0].sum(0).cpu(), cmap='plasma', vmin=0, vmax=3e-6)

        fig.colorbar(im, ax=ax[1,2:])
        

        loss, loss_map_1, _ = get_loss_map(torch.zeros_like(non_makeup), non_makeup, weights1, 1)
        loss, loss_map_2, _ = get_loss_map(torch.zeros_like(non_makeup), non_makeup, weights2, 1)

        im = ax[1,1].imshow((loss_map_1[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma', vmin=0, vmax=2.2)
        fig.colorbar(im, ax=ax[1,:2])
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'wspace':0.0,'hspace':0.0})
        for axis in ax:
            axis.set_axis_off()

        im = ax[0].imshow((loss_map_1[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma', vmin=0, vmax=2.2)
        im = ax[1].imshow((loss_map_2[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma', vmin=0, vmax=2.2)
        fig.colorbar(im, ax=ax[:2])
        plt.show()

        print((loss_map_1[0].sum() / 3).cpu())
        print((loss_map_2[0].sum() / 3).cpu())



        
        fig, ax = plt.subplots(3, 7, gridspec_kw={'wspace':0.0,'hspace':0.05})
        for axis_row in ax:
            for axis in axis_row:
                axis.set_axis_off()

        im = ax[0,0].imshow(weights1[0,0].cpu(), cmap='plasma', vmin=0, vmax=5)
        im = ax[1,0].imshow(non_makeup[0].cpu().permute(1,2,0))
        im = ax[2,0].imshow(transfer_g[0].cpu().permute(1,2,0))

        loss, loss_map_1, loss_maps = get_loss_map(torch.zeros_like(non_makeup), transfer_g, weights1, 5)
        print((loss_map_1[0].sum() / 3).cpu())
        print(loss.cpu())
        im = ax[0,6].imshow((loss_map_1[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma')
        fig.colorbar(im, ax=ax[0,-1])
        for k, loss_map in enumerate(loss_maps):
            im = ax[0,k+1].imshow((loss_map[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma')
        
        loss, loss_map_1, loss_maps = get_loss_map(non_makeup, transfer_g, torch.ones_like(weights1), 5)
        print((loss_map_1[0].sum() / 3).cpu())
        print(loss.cpu())
        im = ax[1,6].imshow((loss_map_1[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma', vmin=0, vmax=0.09)
        fig.colorbar(im, ax=ax[1,-1])
        for k, loss_map in enumerate(loss_maps):
            im = ax[1,k+1].imshow((loss_map[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma')
        
        loss, loss_map_1, loss_maps = get_loss_map(non_makeup, transfer_g, weights1, 5)
        print((loss_map_1[0].sum() / 3).cpu())
        print(loss.cpu())
        im = ax[2,6].imshow((loss_map_1[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma', vmin=0, vmax=0.09)
        fig.colorbar(im, ax=ax[2,-1])
        for k, loss_map in enumerate(loss_maps):
            im = ax[2,k+1].imshow((loss_map[0].sum(0) * 256 * 256 / 3).cpu(), cmap='plasma')
        plt.show()




        




