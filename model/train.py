import torch
from itertools import chain

from option import get_opts
from dataset import create_dataloader
from loss import SAATDLoss, SAATGLoss
from model import get_generator, get_dis_non_makeup, get_dis_makeup


def get_torch_device(opts):
    if opts.platform == 'GPU' and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(opts.device_id))
    else:
        return torch.device('cpu')


def train():
    opts = get_opts()

    device = get_torch_device(opts)

    data_loader = create_dataloader(opts)

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

    for epoch in range(opts.max_epoch):
        d_loss_mean = 0.0
        g_loss_mean = 0.0
        for i, data in enumerate(data_loader):
            h = torch.zeros(1, 1)
            h.to()
            non_makeup = data['non_makeup'].to(device)
            makeup = data['makeup'].to(device)
            transfer_g = data['transfer'].to(device)
            removal_g = data['removal'].to(device)
            non_makeup_parse = data['non_makeup_parse'].to(device)
            makeup_parse = data['makeup_parse'].to(device)

            z_transfer, z_removal, _, _, _, _, _, _ =\
                G(non_makeup, makeup, transfer_g, removal_g, non_makeup_parse, makeup_parse)

            optimizer_D.zero_grad()
            d_loss = loss_D(non_makeup, makeup, z_transfer, z_removal)
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss, _, _, _, _, _, _, _, _ =\
                loss_G(non_makeup, makeup, transfer_g, removal_g, non_makeup_parse, makeup_parse)
            g_loss.backward()
            optimizer_G.step()

            d_loss_mean += d_loss.item()
            g_loss_mean += g_loss.item()
            w_str = ""
            for j in range(i % 10):
                w_str += "."
            print(w_str)

        print("epoch:\t", epoch)
        print("d_loss\t", d_loss_mean / len(data_loader))
        print("g_loss\t", g_loss_mean / len(data_loader))
        print("\n")


if __name__ == "__main__":
    train()
