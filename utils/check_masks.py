import PIL.ImageColor

from dataset_generator.config import get_config
from dataset_generator.training.preprocess import PreProcess
import torch
import os
import numpy as np
from torchvision import transforms
import torchvision

from PIL import Image, ImageDraw

img_dir = 'datasets/test/images/makeup'
mask_dir = 'datasets/test/seg1/makeup'
lms_dir = 'datasets/test/lms/makeup'
config = get_config()
device = torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize(config.DATA.IMG_SIZE),
    transforms.CenterCrop(config.DATA.IMG_SIZE),
    transforms.ToTensor()
])
preprocessor = PreProcess(config, need_parser=False, device=device)

img_names = sorted(os.listdir(img_dir))
green = torch.tensor([[[0]], [[1]], [[0]]], dtype=torch.float).expand(-1, 361, 361)
blue = torch.tensor([[[0]], [[0]], [[1]]], dtype=torch.float).expand(-1, 361, 361)
for img_name in img_names:
    image = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    base_name = os.path.splitext(img_name)[0]
    mask = preprocessor.load_mask(os.path.join(mask_dir, f'{base_name}.png'))
    base_name = os.path.splitext(img_name)[0]
    lms = np.load(os.path.join(lms_dir, f'{base_name}.npy'))
    image = transform(image)

    mask = preprocessor.mask_process(mask)
    """
    image = image * (1 - mask[2:3]) + green * mask[2:3]
    image = image * (1 - mask[3:4]) + blue * mask[3:4]
    """

    image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    points1 = [(lms[i, 1], lms[i, 0]) for i in range(42, 48)]
    points2 = [(lms[i, 1], lms[i, 0]) for i in range(36, 42)]
    """
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
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
            color = image.getpixel((x, y))
            if color == (0, 0, 0):
                ImageDraw.floodfill(image, (x, y), fill_color, thresh=0)
                c += 1
            color = image.getpixel((x_mean, y_mean))
            if color == (0, 0, 0):
                ImageDraw.floodfill(image, (x_mean, y_mean), fill_color, thresh=0)
                c += 1
        print(base_name, c)
        if c == 0:
            draw.polygon(points, fill=fill_color)
    """
    size = 2
    c, _ = lms.shape
    for i in range(c):
        """
        if i in range(14, 17) or i in range(22, 31) or i in range(42, 48):
            fill = 'red'
        else:
            fill = 'black'
        """
        fill = 'red'
        draw.rectangle((lms[i, 1] - size, lms[i, 0] - size, lms[i, 1] + size, lms[i, 0] + size), fill)

    image.save(f'test/{base_name}.png')
