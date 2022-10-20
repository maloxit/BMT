from torchvision.transforms import ToPILImage
import torch

from models.loss import ComposePGT
from training.config import get_config



class PGT_generator():

    def __init__(self, device):
        # Data & PGT
        config = get_config()
        self.img_size = config.DATA.IMG_SIZE
        self.margins = {
            'eye': config.PGT.EYE_MARGIN,
            'lip': config.PGT.LIP_MARGIN
        }
        self.pgt_maker = ComposePGT(
            self.margins,
            config.PGT.SKIN_ALPHA,
            config.PGT.EYE_ALPHA,
            config.PGT.LIP_ALPHA
        )
        self.pgt_maker.eval()

        self.device = device

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
        return fake_A

    def prepare_input(self, *data_inputs, need_batches=False):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            data = data_inputs[i].to(self.device)
            if need_batches:
                data = data.unsqueeze(0)
            inputs.append(data)
        return inputs

    def transfer(self, source_input, reference_input):

        result = self.test(*source_input, *reference_input)

        return result


