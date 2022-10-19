from torchvision.transforms import ToPILImage
import os
import sys
import numpy as np
import torch
import argparse
from itertools import product
from PIL import Image
from tqdm import tqdm
sys.path.append('.')

from training.config import get_config
from training.preprocess import PreProcess

from models.loss import ComposePGT
from datasets import generate_metadata, get_loader



class PGT_generator():

    def __init__(self, config, args):
        self.denoise = config.POSTPROCESS.WILL_DENOISE
        self.preprocess = PreProcess(config, args.device)

        # Data & PGT
        self.img_size = config.DATA.IMG_SIZE
        self.margins = {'eye':config.PGT.EYE_MARGIN,
                        'lip':config.PGT.LIP_MARGIN}
        self.pgt_maker = ComposePGT(self.margins,
            config.PGT.SKIN_ALPHA,
            config.PGT.EYE_ALPHA,
            config.PGT.LIP_ALPHA
        )
        self.pgt_maker.eval()

        self.device = args.device

        super(PGT_generator, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def generate(self, image_A, image_B, mask_A=None, mask_B=None,
                 diff_A=None, diff_B=None, lms_A=None, lms_B=None):
        res = self.pgt_maker(image_A, image_B, mask_A, mask_B, lms_A, lms_B)
        return res

    def test(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B):
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            inputs.append(data_inputs[i].to(self.device))
        return inputs

    def transfer(self, source_input, reference_input, postprocess=True):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """

        result = self.test(*source_input, *reference_input)

        return result


def main(config, args):
    if not args.skip_metadata:
        generate_metadata(args, config, args.device)
    if args.metadata_only:
        return

    generator = PGT_generator(config, args)

    dataloader = get_loader(args, config)

    for data in tqdm(dataloader):
        non_make_up_name = data['non_make_up_name'][0]
        non_makeup = data['non_makeup']
        make_up_name = data['make_up_name'][0]
        makeup = data['makeup']
        non_makeup = generator.prepare_input(*non_makeup)
        makeup = generator.prepare_input(*makeup)
        transfer = generator.transfer(non_makeup, makeup, postprocess=True)
        h = w = config.DATA.IMG_SIZE
        transfer = transfer.resize((h, w))
        transfer = np.array(transfer)
        removal = generator.transfer(makeup, non_makeup, postprocess=True)
        h = w = config.DATA.IMG_SIZE
        removal = removal.resize((h, w))
        removal = np.array(removal)

        non_make_up_name_base = os.path.splitext(non_make_up_name)[0]
        make_up_name_base = os.path.splitext(make_up_name)[0]
        transfer_save_path = os.path.join(args.warp_path, f"{non_make_up_name_base}_{make_up_name_base}.png")
        Image.fromarray(transfer.astype(np.uint8)).save(transfer_save_path)
        removal_save_path = os.path.join(args.warp_path, f"{make_up_name_base}_{non_make_up_name_base}.png")
        Image.fromarray(removal.astype(np.uint8)).save(removal_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--warp-path", type=str, default='result', help="path to warp results")

    parser.add_argument("--non-makeup-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--non-makeup-mask-dir", type=str, default="assets/seg/non-makeup")
    parser.add_argument("--non-makeup-lms-dir", type=str, default="assets/lms/non-makeup")
    parser.add_argument("--makeup-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--makeup-mask-dir", type=str, default="assets/seg/makeup")
    parser.add_argument("--makeup-lms-dir", type=str, default="assets/lms/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")
    parser.add_argument("--skip-metadata", action='store_true', help="Do not generate metadata.")
    parser.add_argument("--metadata-only", action='store_true', help="Only generate metadata.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    if torch.cuda.is_available():
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device('cpu')




    config = get_config()
    main(config, args)
