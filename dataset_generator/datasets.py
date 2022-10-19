import os
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import product

from training.config import get_config
from training.preprocess import PreProcess


def get_loader(args, config):
    dataset = PGTGeneratorDataset(args, config, device=args.device)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS)
    return dataloader

def generate_metadata(args, config, device):
    preprocessor = PreProcess(config, device=device)
    n_img_names = sorted(os.listdir(args.non_makeup_dir))
    m_img_names = sorted(os.listdir(args.makeup_dir))

    if not os.path.exists(args.non_makeup_mask_dir):
        os.makedirs(args.non_makeup_mask_dir)
    if not os.path.exists(args.non_makeup_lms_dir):
        os.makedirs(args.non_makeup_lms_dir)
    if not os.path.exists(args.makeup_mask_dir):
        os.makedirs(args.makeup_mask_dir)
    if not os.path.exists(args.makeup_lms_dir):
        os.makedirs(args.makeup_lms_dir)

    for img_name in n_img_names:
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

    for img_name in m_img_names:
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
        self.non_makeup_dir = args.non_makeup_dir
        self.non_makeup_mask_dir = args.non_makeup_mask_dir
        self.non_makeup_lms_dir = args.non_makeup_lms_dir
        self.makeup_dir = args.makeup_dir
        self.makeup_mask_dir = args.makeup_mask_dir
        self.makeup_lms_dir = args.makeup_lms_dir
        self.n_img_names = sorted(os.listdir(args.non_makeup_dir))
        self.m_img_names = sorted(os.listdir(args.makeup_dir))
        self.preprocessor = PreProcess(config, need_parser=False, device=device)
        self.img_size = config.DATA.IMG_SIZE

    def load_from_file(self, img_name, img_dir, mask_dir, lms_dir):
        image = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(mask_dir, img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(lms_dir, f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def __getitem__(self, index):
        non_make_up_index = index // len(self.m_img_names)
        make_up_index = index % len(self.m_img_names)
        non_make_up_name = self.n_img_names[non_make_up_index]
        make_up_name = self.m_img_names[make_up_index]
        non_makeup = self.load_from_file(non_make_up_name, self.non_makeup_dir, self.non_makeup_mask_dir, self.non_makeup_lms_dir)
        makeup = self.load_from_file(make_up_name, self.makeup_dir, self.makeup_mask_dir, self.makeup_lms_dir)
        data = {
            'non_make_up_name': non_make_up_name,
            'non_makeup': non_makeup,
            'make_up_name': make_up_name,
            'makeup': makeup
        }
        return data

    def __len__(self):
        return len(self.n_img_names) * len(self.m_img_names)

