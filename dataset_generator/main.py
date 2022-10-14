from torchvision.transforms import ToPILImage
import os
import sys
import numpy as np
import torch
import argparse
from itertools import product
from PIL import Image
sys.path.append('.')

from training.config import get_config
from training.preprocess import PreProcess

from models.loss import ComposePGT



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
            inputs.append(data_inputs[i].to(self.device).unsqueeze(0))
        return inputs

    def transfer(self, source: Image, reference: Image, postprocess=True):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)
        result = self.test(*source_input, *reference_input)

        return result


def main(config, args):
    generator = PGT_generator(config, args)
    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))

    if args.comb == 'y':
        iterator = enumerate(product(n_imgname, m_imgname))
    else:
        iterator = enumerate(zip(n_imgname, m_imgname))

    for i, (imga_name, imgb_name) in iterator:
        imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')
        imgB = Image.open(os.path.join(args.reference_dir, imgb_name)).convert('RGB')

        result = generator.transfer(imgA, imgB, postprocess=True)
        if result is None:
            continue
        h = w = config.DATA.IMG_SIZE
        result = result.resize((h, w))
        result = np.array(result)

        if args.result_only == 'y':
            vis_image = result
        else:
            imgA = imgA.resize((h, w))
            imgA = np.array(imgA)
            imgB = imgB.resize((h, w))
            imgB = np.array(imgB)
            vis_image = np.hstack((imgA, imgB, result))

        imga_name = os.path.splitext(imga_name)[0]
        imgb_name = os.path.splitext(imgb_name)[0]
        save_path = os.path.join(args.save_folder, f"{imga_name}_{imgb_name}.png")
        Image.fromarray(vis_image.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model",
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")
    parser.add_argument("--comb", default='y', type=str, help="Test all combinations.")
    parser.add_argument("--result-only", default='y', type=str, help="Output result only.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device('cpu')

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()
    main(config, args)
