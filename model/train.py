import torch
import torchvision
import torch.utils.data as torchdt
from itertools import chain
import os

from model.option import get_opts
from model.dataset import MakeupDataset
from model.loss import SAATDLoss, SAATGLoss
from model.model import get_generator, get_dis_non_makeup, get_dis_makeup

from tqdm import tqdm


def get_torch_device(opts):
    if opts.platform == 'GPU' and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(opts.device_id))
    else:
        return torch.device('cpu')


opts = get_opts()

device = get_torch_device(opts)

# model G and D
G = get_generator(opts, device)
D_non_makeup = get_dis_non_makeup(opts, device)
D_makeup = get_dis_makeup(opts, device)

# loss G and D
loss_G = SAATGLoss(opts, G, D_non_makeup, D_makeup)
loss_D = SAATDLoss(opts, D_non_makeup, D_makeup)

# optimizer G and D
optimizer_G = torch.optim.Adam(G.parameters(), opts.lr, betas=(opts.beta1, opts.beta2), weight_decay=0.0001)
optimizer_D = torch.optim.Adam(chain(D_non_makeup.parameters(), D_makeup.parameters()), opts.lr,
                                betas=(opts.beta1, opts.beta2), weight_decay=0.0001)

last_epoch = -1
pytorch_total_paramsG = sum(p.numel() for p in G.parameters() if p.requires_grad)
pytorch_total_paramsD = sum(p.numel() for p in D_makeup.dis_makeup.parameters() if p.requires_grad)

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

print(numel(G, True))

# Load checkpoint

checkpoint_path = 'outputs/SSAT80.tar'


def load_checkpoint(checkpoint_path, G=None, D_non_makeup=None, D_makeup=None, optimizer_G=None, optimizer_D=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if G is not None:
        G.load_state_dict(checkpoint['model_G_state_dict'])
    if D_non_makeup is not None:
        D_non_makeup.load_state_dict(checkpoint['model_D_non_makeup_state_dict'])
    if D_makeup is not None:
        D_makeup.load_state_dict(checkpoint['model_D_makeup_state_dict'])
    if optimizer_G is not None:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D is not None:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    last_epoch = checkpoint['epoch']
    last_g_loss = checkpoint['g_loss']
    last_d_loss = checkpoint['d_loss']

    print("Loading checkpoint")
    print("epoch:\t", last_epoch)
    print("g_loss\t", last_g_loss)
    print("d_loss\t", last_d_loss)
    print("\n")
    return last_epoch


def write_test_pair_img(result_dir, test_pair_img, iter):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_filename = os.path.join(result_dir, 'gen_%05d.png' % iter)
    torchvision.utils.save_image(test_pair_img / 2 + 0.5, img_filename, nrow=1)


def normalize_image(x):
    return x[:, 0:3, :, :]


def get_img(non_makeup, makeup, z_transfer, z_removal, transfer_g, removal_g):
    # non_makeup_down = normalize_image(F.interpolate(non_makeup, scale_factor=0.25, mode='nearest'))
    # n, c, h, w = non_makeup_down.shape
    # non_makeup_down_warp = torch.bmm(non_makeup_down.view(n, c, h * w), mapY)  # n*HW*1
    # non_makeup_down_warp = non_makeup_down_warp.view(n, c, h, w)
    # non_makeup_warp = F.interpolate(non_makeup_down_warp, scale_factor=4)

    # makeup_down = normalize_image(F.interpolate(makeup, scale_factor=0.25, mode='nearest'))
    # n, c, h, w = makeup_down.shape
    # makeup_down_warp = torch.bmm(makeup_down.view(n, c, h * w), mapX)  # n*HW*1
    # makeup_down_warp = makeup_down_warp.view(n, c, h, w)
    # makeup_warp = F.interpolate(makeup_down_warp, scale_factor=4)

    n, c, h, w = non_makeup.shape

    canvas = torch.ones([1, c, int(h * 2.5), int(w * 4)])

    images_non_makeup = normalize_image(non_makeup).detach()
    images_makeup = normalize_image(makeup).detach()
    images_z_transfer = normalize_image(z_transfer).detach()
    images_z_removal = normalize_image(z_removal).detach()
    images_g_transfer = normalize_image(transfer_g).detach()
    images_g_removal = normalize_image(removal_g).detach()

    canvas[0:1, :, int(h * 0.75): int(h * 0.75) + h, 0:w] = images_non_makeup[0:1, ::]

    canvas[0:1, :, int(h * 0.75): int(h * 0.75) + h, 3 * w:4 * w] = images_makeup[0:1, ::]

    canvas[0:1, :, 0: h, int(w * 1.5):int(w * 1.5) + w] = z_transfer[0:1, ::]

    canvas[0:1, :, int(h * 1.5): int(h * 1.5) + h, int(w * 1.5):int(w * 1.5) + w] = transfer_g[0:1, ::]

    # row1 = torch.cat((images_non_makeup[0:1, ::],images_makeup[0:1, ::], makeup_warp[0:1, ::], images_z_transfer[0:1, ::], images_z_removal[0:1, ::], images_g_transfer[0:1, ::], images_g_removal[0:1, ::]), 3)
    return canvas


dataset = MakeupDataset(opts, device, transform=False, need_pgt=True, all_comb=True)
data_loader = torchdt.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=opts.nThreads)

last_epoch = load_checkpoint(checkpoint_path, G)
G.eval()

import cv2
import numpy as np
from model.loss import GPLoss, CPLoss, WeightMaskGenerator

#gp_loss = GPLoss()
#cp_loss = CPLoss()
#weight_mask_gen = WeightMaskGenerator()

with torch.no_grad():
    for i, data in enumerate(tqdm(data_loader)):
        non_makeup = data['non_makeup'].to(device)
        makeup = data['makeup'].to(device)
        transfer_g = data['transfer'].to(device)
        removal_g = data['removal'].to(device)
        non_makeup_parse = data['non_makeup_parse'].to(device)
        makeup_parse = data['makeup_parse'].to(device)
        """
        for j in range(3):
            img = cv2.imread(f"samples/{i}_{j}.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (opts.crop_size, opts.crop_size))
            img = img / 127.5 - 1.
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0)

            wloss_CP = cp_loss(img, transfer_g, non_makeup_weights)
            wloss_GP = gp_loss(img, transfer_g, non_makeup_weights)
            loss_CP = cp_loss(img, transfer_g, None)
            loss_GP = gp_loss(img, transfer_g, None)
            print(i, j)
            print(loss_CP, wloss_CP)
            print(loss_GP, wloss_GP)
        """
        optimizer_G.zero_grad()
        g_output = G(non_makeup, makeup, non_makeup_parse, makeup_parse)

        z_transfer = g_output[0]
        z_removal = g_output[1]

        #optimizer_D.zero_grad()
        #d_loss = loss_D(non_makeup, makeup, z_transfer.detach(), z_removal.detach())
        #d_loss.backward()
        #optimizer_D.step()

        #g_loss, _, _, _, _, _, _, _, _ =\
        #    loss_G(non_makeup, makeup, transfer_g, removal_g, non_makeup_parse, makeup_parse, g_output, weighted=True)
        #g_loss.backward()
        #optimizer_G.step()
        #img = get_img(non_makeup, makeup, z_transfer, z_removal, transfer_g, removal_g)
        #write_test_pair_img(os.path.join(opts.outputs_dir, 'res{}'.format(last_epoch)), img, i)


