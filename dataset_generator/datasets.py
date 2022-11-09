import os
import shutil
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import argparse
from tqdm import tqdm

from dataset_generator.generator import PGT_generator
from dataset_generator.config import get_config
from dataset_generator.training.preprocess import PreProcess


def fix_mask_eyes(mask, lms, eye_class):
    eye_region = (mask == eye_class[0]).float() + (mask == eye_class[1]).float()
    mask = mask * (1 - eye_region) + 255 * eye_region
    mask_img = transforms.ToPILImage()(mask.squeeze(0).type(torch.ByteTensor))
    draw = ImageDraw.Draw(mask_img)
    lms = lms.numpy()
    points1 = [(lms[i, 1], lms[i, 0]) for i in range(42, 48)]
    points2 = [(lms[i, 1], lms[i, 0]) for i in range(36, 42)]
    color1 = eye_class[0]
    color2 = eye_class[1]
    for fill_color, points in zip((color1, color2), (points1, points2)):
        x_sum = 0
        y_sum = 0
        c = 0
        for i in range(len(points)):
            x, y = points[i]
            x_sum += x
            y_sum += y
            x_mean = x_sum // (i + 1)
            y_mean = y_sum // (i + 1)
            color = mask_img.getpixel((x, y))
            if color == 255:
                ImageDraw.floodfill(mask_img, (x, y), fill_color, thresh=0)
                c += 1
            color = mask_img.getpixel((x_mean, y_mean))
            if color == 255:
                ImageDraw.floodfill(mask_img, (x_mean, y_mean), fill_color, thresh=0)
                c += 1
        if c == 0:
            draw.polygon(points, fill=fill_color)
    return mask_img

def generate_metadata(args, config, device):
    preprocessor = PreProcess(config, device=device)
    n_img_names = sorted(os.listdir(args.non_makeup_dir))
    m_img_names = sorted(os.listdir(args.makeup_dir))
    if n_img_names[0].startswith('.ipynb'):
        n_img_names.pop(0)
    if m_img_names[0].startswith('.ipynb'):
        m_img_names.pop(0)

    if not os.path.exists(args.non_makeup_mask_dir):
        os.makedirs(args.non_makeup_mask_dir)
    if not os.path.exists(args.non_makeup_lms_dir):
        os.makedirs(args.non_makeup_lms_dir)
    if not os.path.exists(args.makeup_mask_dir):
        os.makedirs(args.makeup_mask_dir)
    if not os.path.exists(args.makeup_lms_dir):
        os.makedirs(args.makeup_lms_dir)

    for img_name in tqdm(n_img_names):
        raw_image = Image.open(os.path.join(args.non_makeup_dir, img_name)).convert('RGB')
        base_name = os.path.splitext(img_name)[0]
        mask = preprocessor.face_parse.parse(raw_image)
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (preprocessor.img_size, preprocessor.img_size),
            mode="nearest").squeeze(0).long()  # (1, H, W)
        h, w = raw_image.size
        x_low, y_low, x_high, y_high = (0,0,h,w)
        if h < w:
            y_low = (w - h) // 2
            y_high = w - (w - h + 1) // 2
        else:
            x_low = (h - w) // 2
            x_high = h - (h - w + 1) // 2
        square_image = raw_image.crop((x_low, y_low, x_high, y_high))
        lms = preprocessor.lms_process(square_image)
        mask = fix_mask_eyes(mask, lms, config.PREPROCESS.EYE_CLASS)
        mask.save(os.path.join(args.non_makeup_mask_dir, f'{base_name}.png'))
        preprocessor.save_lms(lms, os.path.join(args.non_makeup_lms_dir, f'{base_name}.npy'))

    for img_name in tqdm(m_img_names):
        raw_image = Image.open(os.path.join(args.makeup_dir, img_name)).convert('RGB')
        base_name = os.path.splitext(img_name)[0]
        mask = preprocessor.face_parse.parse(raw_image)
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (preprocessor.img_size, preprocessor.img_size),
            mode="nearest").squeeze(0).long()  # (1, H, W)
        h, w = raw_image.size
        x_low, y_low, x_high, y_high = (0,0,h,w)
        if h < w:
            y_low = (w - h) // 2
            y_high = w - (w - h + 1) // 2
        else:
            x_low = (h - w) // 2
            x_high = h - (h - w + 1) // 2
        square_image = raw_image.crop((x_low, y_low, x_high, y_high))
        lms = preprocessor.lms_process(square_image)
        mask = fix_mask_eyes(mask, lms, config.PREPROCESS.EYE_CLASS)
        mask.save(os.path.join(args.makeup_mask_dir, f'{base_name}.png'))
        preprocessor.save_lms(lms, os.path.join(args.makeup_lms_dir, f'{base_name}.npy'))


class PGTGeneratorDataset(Dataset):
    def __init__(self, args, config, device):
        self.warp_path = args.warp_path
        self.warp_alt_path = args.warp_alt_path
        self.non_makeup_dir = args.non_makeup_dir
        self.non_makeup_mask_dir = args.non_makeup_mask_dir
        self.non_makeup_lms_dir = args.non_makeup_lms_dir
        self.makeup_dir = args.makeup_dir
        self.makeup_mask_dir = args.makeup_mask_dir
        self.makeup_lms_dir = args.makeup_lms_dir
        self.n_img_names = sorted(os.listdir(args.non_makeup_dir))
        self.m_img_names = sorted(os.listdir(args.makeup_dir))
        if 'skip_to_index' in vars(args).keys():
            self.skip_to_index = args.skip_to_index
        else:
            self.skip_to_index = -1
        if self.n_img_names[0].startswith('.ipynb'):
            self.n_img_names.pop(0)
        if self.m_img_names[0].startswith('.ipynb'):
            self.m_img_names.pop(0)
        self.preprocessor = PreProcess(config, need_parser=False, device=device)
        self.img_size = config.DATA.IMG_SIZE

    def load_from_file(self, img_name, img_dir, mask_dir, lms_dir):
        image = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        base_name = os.path.splitext(img_name)[0]
        mask = self.preprocessor.load_mask(os.path.join(mask_dir, f'{base_name}.png'))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(lms_dir, f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def __getitem__(self, index):
        if index < self.skip_to_index:
            return {'index': index}
        non_make_up_index = index // len(self.m_img_names)
        make_up_index = index % len(self.m_img_names)
        non_make_up_name = self.n_img_names[non_make_up_index]
        make_up_name = self.m_img_names[make_up_index]

        non_make_up_name_base = os.path.splitext(non_make_up_name)[0]
        make_up_name_base = os.path.splitext(make_up_name)[0]

        removal_name = make_up_name_base + '_' + non_make_up_name_base + '.png'
        transfer_name = non_make_up_name_base + '_' + make_up_name_base + '.png'
        modes = ['None', 'transfer', 'removal', 'both']
        mode = 0
        transfer_path = os.path.join(self.warp_alt_path, transfer_name)
        if not os.path.exists(transfer_path):
            transfer_path = os.path.join(self.warp_path, transfer_name)
            if not os.path.exists(transfer_path):
                mode += 1
        removal_path = os.path.join(self.warp_alt_path, removal_name)
        if not os.path.exists(removal_path):
            removal_path = os.path.join(self.warp_path, removal_name)
            if not os.path.exists(removal_path):
                mode += 2
        mode = modes[mode]
        if mode == 'None':
            return {'index': index}
        non_makeup = self.load_from_file(non_make_up_name, self.non_makeup_dir, self.non_makeup_mask_dir, self.non_makeup_lms_dir)
        makeup = self.load_from_file(make_up_name, self.makeup_dir, self.makeup_mask_dir, self.makeup_lms_dir)
        data = {
            'index': index,
            'non_make_up_name': non_make_up_name,
            'non_makeup': non_makeup,
            'make_up_name': make_up_name,
            'makeup': makeup,
            'generate_mode': mode
        }
        return data

    def __len__(self):
        return len(self.n_img_names) * len(self.m_img_names)


class GeneratorManager:
    def __init__(self, args, device):
        self.config = get_config()
        self.warp_path = args.warp_path
        self.warp_alt_path = args.warp_alt_path
        self.warp_storage = args.warp_storage
        if 'storage_every' in vars(args).keys():
            self.storage_every = args.storage_every
        else:
            self.storage_every = 1000
        if not os.path.exists(args.warp_path):
            os.makedirs(args.warp_path)
        if not os.path.exists(args.warp_alt_path):
            os.makedirs(args.warp_alt_path)
        if not os.path.exists(args.warp_storage):
            os.makedirs(args.warp_storage)

        self.generator = PGT_generator(device)
        self.dataset = PGTGeneratorDataset(args, self.config, device=device)

    def move_to_storage(self):
        warp_names = os.listdir(self.warp_path)
        for warp_name in warp_names:
            if warp_name.startswith('.ipynb'):
                continue
            shutil.copy(os.path.join(self.warp_path, warp_name), os.path.join(self.warp_alt_path, warp_name))
            shutil.move(os.path.join(self.warp_path, warp_name), os.path.join(self.warp_storage, warp_name))

    def generate_dataset(self):
        dataloader = DataLoader(dataset=self.dataset,
                                batch_size=1,
                                num_workers=1)
        for data in tqdm(dataloader):
            if data['index'][0] % self.storage_every == 0:
                self.move_to_storage()
            if len(data) == 1:
                continue
            non_make_up_name = data['non_make_up_name'][0]
            non_makeup = data['non_makeup']
            make_up_name = data['make_up_name'][0]
            makeup = data['makeup']
            generate_mode = data['generate_mode'][0]
            non_makeup = self.generator.prepare_input(*non_makeup)
            makeup = self.generator.prepare_input(*makeup)

            non_make_up_name_base = os.path.splitext(non_make_up_name)[0]
            make_up_name_base = os.path.splitext(make_up_name)[0]
            if generate_mode in ('both', 'transfer'):
                transfer = self.generator.transfer(non_makeup, makeup)
                transfer_save_path = os.path.join(self.warp_path, f"{non_make_up_name_base}_{make_up_name_base}.png")
                save_image(transfer, transfer_save_path)
            if generate_mode in ('both', 'removal'):
                removal = self.generator.transfer(makeup, non_makeup)
                removal_save_path = os.path.join(self.warp_path, f"{make_up_name_base}_{non_make_up_name_base}.png")
                save_image(removal, removal_save_path)

    def generate(self, non_make_up_name, make_up_name, mode='both'):
        non_makeup = self.dataset.load_from_file(
            non_make_up_name,
            self.dataset.non_makeup_dir,
            self.dataset.non_makeup_mask_dir,
            self.dataset.non_makeup_lms_dir
        )
        makeup = self.dataset.load_from_file(
            make_up_name,
            self.dataset.makeup_dir,
            self.dataset.makeup_mask_dir,
            self.dataset.makeup_lms_dir
        )
        non_makeup = self.generator.prepare_input(*non_makeup, need_batches=True)
        makeup = self.generator.prepare_input(*makeup, need_batches=True)
        non_make_up_name_base = os.path.splitext(non_make_up_name)[0]
        make_up_name_base = os.path.splitext(make_up_name)[0]
        if mode in ('both', 'transfer'):
            transfer = self.generator.transfer(non_makeup, makeup)
            transfer_save_path = os.path.join(self.warp_path, f"{non_make_up_name_base}_{make_up_name_base}.png")
            save_image(transfer, transfer_save_path)
        if mode in ('both', 'removal'):
            removal = self.generator.transfer(makeup, non_makeup)
            removal_save_path = os.path.join(self.warp_path, f"{make_up_name_base}_{non_make_up_name_base}.png")
            save_image(removal, removal_save_path)


def run():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--warp-path", type=str, default='datasets/train/images/wrap_tmp', help="path to warp results")
    parser.add_argument("--warp-alt-path", type=str, default='datasets/train/images/wrap', help="path to warp results")
    parser.add_argument("--warp-storage", type=str, default='datasets/train/images/wrap_storage')
    parser.add_argument("--storage-every", type=int, default=600)
    parser.add_argument("--skip-to-index", type=int, default=-1)

    parser.add_argument("--non-makeup-dir", type=str, default="datasets/train/images/non-makeup")
    parser.add_argument("--non-makeup-mask-dir", type=str, default="datasets/train/seg1/non-makeup")
    parser.add_argument("--non-makeup-lms-dir", type=str, default="datasets/train/lms/non-makeup")
    parser.add_argument("--makeup-dir", type=str, default="datasets/train/images/makeup")
    parser.add_argument("--makeup-mask-dir", type=str, default="datasets/train/seg1/makeup")
    parser.add_argument("--makeup-lms-dir", type=str, default="datasets/train/lms/makeup")
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
    if not args.skip_metadata:
        generate_metadata(args, config, args.device)
    if not args.metadata_only:
        generator_manager = GeneratorManager(args, args.device)
        generator_manager.generate_dataset()
