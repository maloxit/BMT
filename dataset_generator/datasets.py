import os
import shutil
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import argparse
from tqdm import tqdm

from dataset_generator.generator import PGT_generator
from dataset_generator.config import get_config
from dataset_generator.training.preprocess import PreProcess


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

        np_image = np.array(raw_image)
        mask = preprocessor.face_parse.parse(cv2.resize(np_image, (512, 512)))
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (preprocessor.img_size, preprocessor.img_size),
            mode="nearest").squeeze(0).long()  # (1, H, W)
        preprocessor.save_mask(mask, os.path.join(args.non_makeup_mask_dir, img_name))

        lms = preprocessor.lms_process(raw_image)
        base_name = os.path.splitext(img_name)[0]
        preprocessor.save_lms(lms, os.path.join(args.non_makeup_lms_dir, f'{base_name}.npy'))

    for img_name in tqdm(m_img_names):
        raw_image = Image.open(os.path.join(args.makeup_dir, img_name)).convert('RGB')

        np_image = np.array(raw_image)
        mask = preprocessor.face_parse.parse(cv2.resize(np_image, (512, 512)))
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (preprocessor.img_size, preprocessor.img_size),
            mode="nearest").squeeze(0).long()  # (1, H, W)
        preprocessor.save_mask(mask, os.path.join(args.makeup_mask_dir, img_name))

        lms = preprocessor.lms_process(raw_image)
        base_name = os.path.splitext(img_name)[0]
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
        self.skip_to_index = args.skip_to_index
        if self.n_img_names[0].startswith('.ipynb'):
            self.n_img_names.pop(0)
        if self.m_img_names[0].startswith('.ipynb'):
            self.m_img_names.pop(0)
        self.preprocessor = PreProcess(config, need_parser=False, device=device)
        self.img_size = config.DATA.IMG_SIZE

    def load_from_file(self, img_name, img_dir, mask_dir, lms_dir):
        image = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(mask_dir, img_name))
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
        modes = [None, 'transfer', 'removal', 'both']
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
        self.storage_every = args.storage_every
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
    parser.add_argument("--warp-path", type=str, default='result', help="path to warp results")
    parser.add_argument("--warp-alt-path", type=str, default='result', help="path to warp results")
    parser.add_argument("--warp-storage", type=str, default='result_storage')
    parser.add_argument("--storage-every", type=int, default=600)
    parser.add_argument("--skip-to-index", type=int, default=-1)

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
    if not args.skip_metadata:
        generate_metadata(args, config, args.device)
    if not args.metadata_only:
        generator_manager = GeneratorManager(args, args.device)
        generator_manager.generate_dataset()
