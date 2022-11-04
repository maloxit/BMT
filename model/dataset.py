import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

from dataset_generator.datasets import GeneratorManager


class MakeupDataset(data.Dataset):
    """import dataset"""

    def __init__(self, opts, device, *, transform=True, need_pgt=True, all_comb=False):
        """init"""
        self.transform = transform
        self.need_pgt = need_pgt
        self.all_comb = all_comb
        self.opt = opts
        self.phase = opts.phase
        self.semantic_dim = opts.semantic_dim
        self.non_makeup_dir = opts.non_makeup_dir
        self.non_makeup_mask_dir = opts.non_makeup_mask_dir
        self.makeup_dir = opts.makeup_dir
        self.makeup_mask_dir = opts.makeup_mask_dir
        self.warp_path = opts.warp_path
        if not os.path.exists(self.warp_path):
            os.makedirs(self.warp_path)
        self.warp_alt_path = opts.warp_alt_path
        if not os.path.exists(self.warp_alt_path):
            os.makedirs(self.warp_alt_path)
        self.warp_storage = opts.warp_storage
        if not os.path.exists(self.warp_storage):
            os.makedirs(self.warp_storage)

        self.generator = GeneratorManager(opts, device)

        # non_makeup
        self.name_non_makeup = os.listdir(self.non_makeup_dir)

        # makeup
        self.name_makeup = os.listdir(self.makeup_dir)

        self.non_makeup_size = len(self.name_non_makeup)
        self.makeup_size = len(self.name_makeup)
        print('non_makeup size:', self.non_makeup_size, 'makeup size:', self.makeup_size)
        if self.all_comb:
            self.dataset_size = self.non_makeup_size * self.makeup_size
        else:
            self.dataset_size = self.non_makeup_size

    def load_img(self, img_path, angle=0):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.rotate(img, angle)
        return img

    def load_parse(self, parse, angle=0):
        parse = cv2.imread(parse, cv2.IMREAD_GRAYSCALE)
        parse = self.rotate(parse, angle)
        h, w = parse.shape
        result = np.zeros([h, w, self.semantic_dim])
        for i in range(self.semantic_dim):
            result[:, :, i][np.where(parse == i)] = 1
        result = np.array(result)
        return result

    def rotate(self, img, angle):
        img = Image.fromarray(img)
        img = img.rotate(angle)
        img = np.array(img)

        return img

    def move_warp_to_storage(self):
        warp_names = os.listdir(self.warp_path)
        for warp_name in warp_names:
            if warp_name.startswith('.ipynb'):
                continue
            shutil.copy(os.path.join(self.warp_path, warp_name), os.path.join(self.warp_alt_path, warp_name))
            shutil.move(os.path.join(self.warp_path, warp_name), os.path.join(self.warp_storage, warp_name))

    def __getitem__(self, index):
        if self.transform and np.random.random() > 0.5:
            non_makeup_angle = np.random.randint(0, 60) - 30
            makeup_angle = np.random.randint(0, 60) - 30
        else:
            non_makeup_angle = 0
            makeup_angle = 0

        if self.all_comb:
            non_makeup_index = index // self.makeup_size
            makeup_index = index % self.makeup_size
        else:
            non_makeup_index = index
            makeup_index = random.randint(0, self.makeup_size - 1)
        

        non_makeup_img = self.load_img(os.path.join(self.non_makeup_dir, self.name_non_makeup[non_makeup_index]),
                                       non_makeup_angle)
        makeup_img = self.load_img(os.path.join(self.makeup_dir, self.name_makeup[makeup_index]), makeup_angle)
        
        non_makeup_name = os.path.splitext(self.name_non_makeup[non_makeup_index])[0]
        makeup_name = os.path.splitext(self.name_makeup[makeup_index])[0]

        non_makeup_parse = self.load_parse(
            os.path.join(self.non_makeup_mask_dir, f'{non_makeup_name}.npy'), non_makeup_angle)
        makeup_parse = self.load_parse(os.path.join(self.makeup_mask_dir, f'{makeup_name}.npy'), makeup_angle)

        # load groundtrue
        transfer_img = None
        removal_img = None
        if self.need_pgt:
            removal_name = makeup_name + '_' + non_makeup_name + '.png'
            transfer_name = non_makeup_name + '_' + makeup_name + '.png'
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
            if mode is not None:
                self.generator.generate(self.name_non_makeup[non_makeup_index], self.name_makeup[makeup_index],
                                        mode=mode)
            transfer_img = self.load_img(transfer_path, non_makeup_angle)
            removal_img = self.load_img(removal_path, makeup_angle)

        # preprocessing
        non_makeup_img, makeup_img, non_makeup_parse, makeup_parse, transfer_img, removal_img = self.preprocessing(
            opts=self.opt, non_makeup_img=non_makeup_img, makeup_img=makeup_img,
            non_makeup_parse=non_makeup_parse, makeup_parse=makeup_parse, transfer_img=transfer_img,
            removal_img=removal_img)

        non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1)).astype(np.float32)
        makeup_img = np.transpose(makeup_img, (2, 0, 1)).astype(np.float32)
        non_makeup_parse = np.transpose(non_makeup_parse, (2, 0, 1)).astype(np.float32)
        makeup_parse = np.transpose(makeup_parse, (2, 0, 1)).astype(np.float32)
        non_makeup_parse = np.clip(non_makeup_parse, a_min=0, a_max=1).astype(np.float32)
        makeup_parse = np.clip(makeup_parse, a_min=0, a_max=1).astype(np.float32)

        data = {
            'non_makeup': non_makeup_img, 'makeup': makeup_img,
            'non_makeup_parse': non_makeup_parse, 'makeup_parse': makeup_parse
        }

        if self.need_pgt:
            transfer_img = np.transpose(transfer_img, (2, 0, 1)).astype(np.float32)
            removal_img = np.transpose(removal_img, (2, 0, 1)).astype(np.float32)
            data['transfer'] = transfer_img
            data['removal'] = removal_img

        return data

    def __len__(self):
        return self.dataset_size

    def preprocessing(self, opts, non_makeup_img, makeup_img, non_makeup_parse, makeup_parse,
                      transfer_img=None, removal_img=None):
        if self.transform:
            if np.random.random() > 0.5:
                non_makeup_img = cv2.resize(non_makeup_img, (opts.resize_size, opts.resize_size))
                non_makeup_parse = cv2.resize(non_makeup_parse, (opts.resize_size, opts.resize_size),
                                              interpolation=cv2.INTER_NEAREST)
                h1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
                w1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
                non_makeup_img = non_makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
                non_makeup_parse = non_makeup_parse[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
                if self.need_pgt:
                    transfer_img = cv2.resize(transfer_img, (opts.resize_size, opts.resize_size))
                    transfer_img = transfer_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
            if np.random.random() > 0.5:
                makeup_img = cv2.resize(makeup_img, (opts.resize_size, opts.resize_size))
                makeup_parse = cv2.resize(makeup_parse, (opts.resize_size, opts.resize_size),
                                          interpolation=cv2.INTER_NEAREST)
                h1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
                w1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
                makeup_img = makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
                makeup_parse = makeup_parse[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
                if self.need_pgt:
                    removal_img = cv2.resize(removal_img, (opts.resize_size, opts.resize_size))
                    removal_img = removal_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]

            if opts.flip:
                if np.random.random() > 0.5:
                    non_makeup_img = np.fliplr(non_makeup_img)
                    makeup_img = np.fliplr(makeup_img)
                    non_makeup_parse = np.fliplr(non_makeup_parse)
                    makeup_parse = np.fliplr(makeup_parse)
                    if self.need_pgt:
                        transfer_img = np.fliplr(transfer_img)
                        removal_img = np.fliplr(removal_img)

        non_makeup_img = cv2.resize(non_makeup_img, (opts.crop_size, opts.crop_size))
        makeup_img = cv2.resize(makeup_img, (opts.crop_size, opts.crop_size))
        non_makeup_parse = cv2.resize(non_makeup_parse, (opts.crop_size, opts.crop_size),
                                      interpolation=cv2.INTER_NEAREST)
        makeup_parse = cv2.resize(makeup_parse, (opts.crop_size, opts.crop_size), interpolation=cv2.INTER_NEAREST)
        if self.need_pgt:
            transfer_img = cv2.resize(transfer_img, (opts.crop_size, opts.crop_size))
            removal_img = cv2.resize(removal_img, (opts.crop_size, opts.crop_size))

        non_makeup_img = non_makeup_img / 127.5 - 1.
        makeup_img = makeup_img / 127.5 - 1.
        if self.need_pgt:
            transfer_img = transfer_img / 127.5 - 1.
            removal_img = removal_img / 127.5 - 1.
        data = {'non_makeup': non_makeup_img, 'makeup': makeup_img, 'transfer': transfer_img, 'removal': removal_img,
                'non_makeup_parse': non_makeup_parse, 'makeup_parse': makeup_parse}
        return non_makeup_img, makeup_img, non_makeup_parse, makeup_parse, transfer_img, removal_img
