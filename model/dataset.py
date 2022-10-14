import os
import cv2
import copy
import random
import numpy as np
from PIL import Image
import torch.utils.data as data


def create_dataloader(opts):
    dataset = MakeupDataset(opts)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opts.nThreads)
    return dataloader


class MakeupDataset(data.Dataset):
    """import dataset"""

    def __init__(self, opts):
        """init"""
        self.opt = opts
        self.phase = opts.phase
        self.dataroot = opts.dataroot
        self.semantic_dim = opts.semantic_dim

        # non_makeup
        name_non_makeup = os.listdir(os.path.join(self.dataroot, 'non-makeup'))
        self.non_makeup_path = [os.path.join(self.dataroot, 'non-makeup', x) for x in name_non_makeup]

        # makeup
        name_makeup = os.listdir(os.path.join(self.dataroot, 'makeup'))
        self.makeup_path = [os.path.join(self.dataroot, 'makeup', x) for x in name_makeup]

        self.warproot = os.path.join(self.dataroot, 'warp')

        self.non_makeup_size = len(self.non_makeup_path)
        self.makeup_size = len(self.makeup_path)
        print('non_makeup size:', self.non_makeup_size, 'makeup size:', self.makeup_size)
        if self.phase == 'train':
            self.dataset_size = self.non_makeup_size * self.makeup_size
        else:
            self.dataset_size = self.non_makeup_size * self.makeup_size

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

    def __getitem__(self, index):
        if self.phase == 'train':
            if np.random.random() > 0.5:
                non_makeup_angle = np.random.randint(0, 60) - 30
                makeup_angle = np.random.randint(0, 60) - 30
            else:
                non_makeup_angle = 0
                makeup_angle = 0
            non_makeup_index = index // self.makeup_size
            makeup_index = index % self.makeup_size
            # print('non_makeup_angle',non_makeup_angle,'makeup_angle',makeup_angle)
            # load non-makeup
            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index], non_makeup_angle)
            non_makeup_mask = self.load_img(self.non_makeup_path[non_makeup_index].replace('images', 'seg1'), non_makeup_angle)
            non_makeup_parse = self.load_parse(self.non_makeup_path[non_makeup_index].replace('images', 'seg1'), non_makeup_angle)

            # load makeup
            index_other = random.randint(0, self.makeup_size - 1)
            makeup_img = self.load_img(self.makeup_path[makeup_index], makeup_angle)
            makeup_mask = self.load_img(self.makeup_path[makeup_index].replace('images', 'seg1'), makeup_angle)
            makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('images', 'seg1'), makeup_angle)

            # load groundtrue
            non_makeup_name = os.path.basename(self.non_makeup_path[non_makeup_index])[:-4]
            makeup_name = os.path.basename(self.makeup_path[makeup_index])[:-4]
            transfer_name = makeup_name + '_' + non_makeup_name + '.png'
            removal_name = non_makeup_name + '_' + makeup_name + '.png'
            transfer_img = self.load_img(os.path.join(self.warproot, transfer_name))
            removal_img = self.load_img(os.path.join(self.warproot, removal_name))
            h, w, c = transfer_img.shape
            #transfer_img = transfer_img[:, 2 * h:3 * h, :]
            #removal_img = removal_img[:, 2 * h:3 * h, :]
            transfer_img = self.rotate(transfer_img, non_makeup_angle)
            removal_img = self.rotate(removal_img, makeup_angle)

            # preprocessing
            data = self.preprocessing(opts=self.opt, non_makeup_img=non_makeup_img, makeup_img=makeup_img,
                                      transfer_img=transfer_img, removal_img=removal_img,
                                      non_makeup_mask=non_makeup_mask, makeup_mask=makeup_mask,
                                      non_makeup_parse=non_makeup_parse, makeup_parse=makeup_parse)

            non_makeup_img = data['non_makeup']
            makeup_img = data['makeup']
            transfer_img = data['transfer']
            removal_img = data['removal']
            non_makeup_parse = data['non_makeup_parse']
            makeup_parse = data['makeup_parse']

            non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1)).astype(np.float32)
            makeup_img = np.transpose(makeup_img, (2, 0, 1)).astype(np.float32)
            transfer_img = np.transpose(transfer_img, (2, 0, 1)).astype(np.float32)
            removal_img = np.transpose(removal_img, (2, 0, 1)).astype(np.float32)
            non_makeup_parse = np.transpose(non_makeup_parse, (2, 0, 1)).astype(np.float32)
            makeup_parse = np.transpose(makeup_parse, (2, 0, 1)).astype(np.float32)
            non_makeup_parse = np.clip(non_makeup_parse, a_min=0, a_max=1).astype(np.float32)
            makeup_parse = np.clip(makeup_parse, a_min=0, a_max=1).astype(np.float32)

            data = {
                'non_makeup': non_makeup_img, 'makeup': makeup_img,
                'transfer': transfer_img, 'removal': removal_img,
                'non_makeup_parse': non_makeup_parse, 'makeup_parse': makeup_parse
            }

            return data
        else:
            non_makeup_index = index // self.makeup_size
            makeup_index = index % self.makeup_size
            # non_makeup_index = index
            # makeup_index = index
            print(self.non_makeup_size, self.makeup_size, non_makeup_index + 1, makeup_index + 1)

            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index])
            non_makeup_mask = self.load_img(self.non_makeup_path[non_makeup_index].replace('images', 'seg1'))
            non_makeup_parse = self.load_parse(self.non_makeup_path[non_makeup_index].replace('images', 'seg1'))

            # load makeup
            makeup_img = self.load_img(self.makeup_path[makeup_index])
            makeup_mask = self.load_img(self.makeup_path[makeup_index].replace('images', 'seg1'))
            makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('images', 'seg1'))

            # preprocessing
            data = self.preprocessing(opts=self.opt, non_makeup_img=non_makeup_img, makeup_img=makeup_img,
                                      transfer_img=non_makeup_img, removal_img=makeup_img,
                                      non_makeup_mask=non_makeup_mask, makeup_mask=makeup_mask,
                                      non_makeup_parse=non_makeup_parse, makeup_parse=makeup_parse)

            non_makeup_img = data['non_makeup']
            makeup_img = data['makeup']
            non_makeup_parse = data['non_makeup_parse']
            makeup_parse = data['makeup_parse']

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

            return data

    def __len__(self):
        return self.dataset_size

    def expand_mask(self, mask):
        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate((mask, mask, mask), axis=2)
        return mask

    def preprocessing(self, opts, non_makeup_img, makeup_img, transfer_img, removal_img, non_makeup_mask, makeup_mask,
                      non_makeup_parse, makeup_parse):
        non_makeup_img = cv2.resize(non_makeup_img, (opts.resize_size, opts.resize_size))
        makeup_img = cv2.resize(makeup_img, (opts.resize_size, opts.resize_size))

        transfer_img = cv2.resize(transfer_img, (opts.resize_size, opts.resize_size))
        removal_img = cv2.resize(removal_img, (opts.resize_size, opts.resize_size))

        non_makeup_mask = cv2.resize(non_makeup_mask, (opts.resize_size, opts.resize_size),
                                     interpolation=cv2.INTER_NEAREST)
        makeup_mask = cv2.resize(makeup_mask, (opts.resize_size, opts.resize_size),
                                 interpolation=cv2.INTER_NEAREST)

        non_makeup_parse = cv2.resize(non_makeup_parse, (opts.resize_size, opts.resize_size),
                                      interpolation=cv2.INTER_NEAREST)
        makeup_parse = cv2.resize(makeup_parse, (opts.resize_size, opts.resize_size),
                                  interpolation=cv2.INTER_NEAREST)

        transfer_img = self.get_groundtrue(transfer_img, non_makeup_mask, transfer_img, non_makeup_mask)
        removal_img = self.get_groundtrue(removal_img, makeup_mask, removal_img, makeup_mask)

        if np.random.random() > 0.5:
            h1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
            non_makeup_img = non_makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
            transfer_img = transfer_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
            non_makeup_parse = non_makeup_parse[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
        if np.random.random() > 0.5:
            h1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, opts.resize_size - opts.crop_size)))
            makeup_img = makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
            removal_img = removal_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
            makeup_parse = makeup_parse[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]

        if opts.flip:
            if np.random.random() > 0.5:
                non_makeup_img = np.fliplr(non_makeup_img)
                makeup_img = np.fliplr(makeup_img)
                transfer_img = np.fliplr(transfer_img)
                removal_img = np.fliplr(removal_img)
                non_makeup_parse = np.fliplr(non_makeup_parse)
                makeup_parse = np.fliplr(makeup_parse)

        non_makeup_img = cv2.resize(non_makeup_img, (opts.crop_size, opts.crop_size))
        makeup_img = cv2.resize(makeup_img, (opts.crop_size, opts.crop_size))
        transfer_img = cv2.resize(transfer_img, (opts.crop_size, opts.crop_size))
        removal_img = cv2.resize(removal_img, (opts.crop_size, opts.crop_size))
        non_makeup_parse = cv2.resize(non_makeup_parse, (opts.crop_size, opts.crop_size),
                                      interpolation=cv2.INTER_NEAREST)
        makeup_parse = cv2.resize(makeup_parse, (opts.crop_size, opts.crop_size), interpolation=cv2.INTER_NEAREST)

        non_makeup_img = non_makeup_img / 127.5 - 1.
        makeup_img = makeup_img / 127.5 - 1.
        transfer_img = transfer_img / 127.5 - 1.
        removal_img = removal_img / 127.5 - 1.
        data = {'non_makeup': non_makeup_img, 'makeup': makeup_img, 'transfer': transfer_img, 'removal': removal_img,
                'non_makeup_parse': non_makeup_parse, 'makeup_parse': makeup_parse}
        return data

    def get_groundtrue(self, source_img, source_mask, reference_img, reference_mask):
        source_mask_neck = self.get_neck_ear_mask(copy.copy(source_mask))

        reference_mask_neck = self.get_face_mask(copy.copy(reference_mask))

        source_img_neck = source_img * source_mask_neck

        reference_img_neck = reference_img * reference_mask_neck

        h, w, c = source_img.shape
        groundtrue_neck = self.hist_match_func(source_img_neck, reference_img_neck)

        groundtrue_neck = np.reshape(groundtrue_neck, [h, w, c])

        source_img[np.where(source_mask_neck == 1)] = groundtrue_neck[np.where(source_mask_neck == 1)]

        return source_img

    # get neck and ear mask
    def get_neck_ear_mask(self, mask):
        mask[np.where(mask == 1)] = 0
        mask[np.where(mask == 14)] = 1
        mask[np.where(mask == 8)] = 1
        mask[np.where(mask == 7)] = 1
        mask[np.where(mask != 1)] = 0
        return mask

    # get face mask
    def get_face_mask(self, mask):
        mask[np.where(mask != 1)] = 0
        # temp = np.zeros_like(mask)
        # temp[np.where(mask != 1)] = 0
        return mask

    # 直方图匹配
    def hist_match_func(self, source, reference):
        """
        Adjust the pixel values of images such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            reference: np.ndarray
                Reference image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """
        source = np.expand_dims(source, axis=0)
        reference = np.expand_dims(reference, axis=0)

        oldshape = source.shape
        batch_size = oldshape[0]
        source = np.array(source, dtype=np.uint8)
        reference = np.array(reference, dtype=np.uint8)
        # get the set of unique pixel values and their corresponding indices and
        # counts
        result = np.zeros(oldshape, dtype=np.uint8)
        for i in range(batch_size):
            for c in range(3):
                s = source[i, ..., c].ravel()
                r = reference[i, ..., c].ravel()

                s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
                r_values, r_counts = np.unique(r, return_counts=True)

                if (len(s_counts) == 1 or len(r_counts) == 1):
                    continue
                # take the cumsum of the counts and normalize by the number of pixels to
                # get the empirical cumulative distribution functions for the source and
                # template images (maps pixel value --> quantile)
                s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
                s_quantiles /= s_quantiles[-1]
                r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
                r_quantiles /= r_quantiles[-1]
                r_values = r_values[1:]

                # interpolate linearly to find the pixel values in the template image
                # that correspond most closely to the quantiles in the source image
                interp_value = np.zeros_like(s_values, dtype=np.float32)
                interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
                interp_value[1:] = interp_r_values
                result[i, ..., c] = interp_value[bin_idx].reshape(oldshape[1:3])
        result = np.array(result, dtype=np.float32)
        return result
