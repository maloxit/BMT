import argparse


def get_opts():
    parser = MakeupOptions()
    opts = parser.parse()
    return opts


class MakeupOptions:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument("--warp-path", type=str, default='result', help="path to warp results")
        self.parser.add_argument("--warp-alt-path", type=str, default='result_alt')
        self.parser.add_argument("--warp-storage", type=str, default='result_storage')

        self.parser.add_argument("--non-makeup-dir", type=str, default="assets/images/non-makeup")
        self.parser.add_argument("--non-makeup-mask-dir", type=str, default="assets/seg/non-makeup")
        self.parser.add_argument("--non-makeup-lms-dir", type=str, default="assets/lms/non-makeup")
        self.parser.add_argument("--makeup-dir", type=str, default="assets/images/makeup")
        self.parser.add_argument("--makeup-mask-dir", type=str, default="assets/seg/makeup")
        self.parser.add_argument("--makeup-lms-dir", type=str, default="assets/lms/makeup")

        self.parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
        self.parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
        self.parser.add_argument('--semantic_dim', type=int, default=18, help='output_dim')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--flip', type=bool, default=True, help='specified if  flipping')
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')

        # platform related
        self.parser.add_argument('--platform', type=str, default='GPU', help='only support GPU and CPU')
        self.parser.add_argument('--device_id', type=int, default=0, help='device id, default is 0.')
        self.parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='SSAT', help='folder name to save outputs')
        self.parser.add_argument('--outputs_dir', type=str, default='./outputs',
                                 help='models are saved here, default is ./outputs.')
        self.parser.add_argument('--print_iter', type=int, default=100, help='log print iter, default is 100.')
        self.parser.add_argument('--save_imgs', type=bool, default=True, help='whether save imgs when epoch end')
        self.parser.add_argument('--save_checkpoint_epochs', type=int, default=100,
                                 help='save_checkpoint_epochs, default is 100.')

        # weight
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', help='gan_mode')
        self.parser.add_argument('--rec_weight', type=float, default=1, help='rec_weight')
        self.parser.add_argument('--CP_weight', type=float, default=2, help='CP_weight')
        self.parser.add_argument('--GP_weight', type=float, default=1, help='CP_weight')
        self.parser.add_argument('--cycle_weight', type=float, default=1, help='cycle_weight')
        self.parser.add_argument('--adv_weight', type=float, default=1, help='adv_weight')
        self.parser.add_argument('--latent_weight', type=float, default=0.1, help='latent_weight')
        self.parser.add_argument('--semantic_weight', type=float, default=1, help='semantic_weight')

        # training related
        self.parser.add_argument('--init_type', type=str, default='normal', help='init_type')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='init_gain')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2')

        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', type=bool, default=True,
                                 help='use spectral normalization in discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')

        self.parser.add_argument('--max_epoch', type=int, default=1000, help='epoch size for training, default is 200.')
        self.parser.add_argument('--n_epochs', type=int, default=1000,
                                 help='number of epochs with the initial learning rate, default is 100')
        self.parser.add_argument('--n_epochs_decay', type=int, default=500,
                                 help='n_epochs_decay')

        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--num_residule_block', type=int, default=4, help='num_residule_block')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='lr')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
