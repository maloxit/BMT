import telebot
import torch
import torchvision
import torch.utils.data as torchdt
import os

from model.dataset import MakeupDataset
from model.model import get_generator
from dataset_generator.datasets import generate_metadata
import argparse
import queue
from tqdm import tqdm
import threading

class myThread (threading.Thread):
    def __init__(self, request_queue: queue.Queue, request_handler):
        threading.Thread.__init__(self)
        self.request_queue = request_queue
        self.request_handler = request_handler
    def run(self):
        while True:
            request = self.request_queue.get()
            if request is None:
                return
            self.request_handler(request)

def get_torch_device(opts):
    if opts.platform == 'GPU' and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(opts.device_id))
    else:
        return torch.device('cpu')


def load_checkpoint(checkpoint_path, device, G=None, D_non_makeup=None, D_makeup=None, optimizer_G=None, optimizer_D=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if G is not None:
        G.load_state_dict(checkpoint['model_G_state_dict'])
    if D_non_makeup is not None:
        D_non_makeup.load_state_dict(checkpoint['model_D_non_makeup_state_dict'])
    if D_makeup is not None:
        D_makeup.load_state_dict(checkpoint['model_D_makeup_state_dict'])
    if optimizer_G is not None:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D is not None:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    last_epoch = checkpoint['epoch']
    last_g_loss = checkpoint['g_loss']
    last_d_loss = checkpoint['d_loss']

    print("Loading checkpoint")
    print("epoch:\t", last_epoch)
    print("g_loss\t", last_g_loss)
    print("d_loss\t", last_d_loss)
    print("\n")
    return last_epoch

class Bot(telebot.TeleBot):
    def __init__(self, token: str, checkpoint_path: str):
        super().__init__(token=token, threaded=False, skip_pending=True)
        self.request_queue = queue.Queue(5)

        self.opts = argparse.Namespace(**{
            'warp_dir': 'bot/tmp/images/warp',
            'warp_alt_dir': 'bot/tmp/images/warp',
            'warp_storage_dir': 'bot/tmp/images/warp',
            'subset_config_files': 
            [
                'bot/bot_test_makeup.json',
                'bot/bot_non-makeup.json'
            ],
            'input_dim': 3,
            'output_dim': 3,
            'semantic_dim': 18,
            'batch_size': 1,
            'resize_size': 286,
            'crop_size': 256,
            'flip': True,
            'nThreads': 2,

            # platform related
            'platform': 'GPU',
            'device_id': 0,
            'device_num': 1,

            # ouptput related
            'name': 'BMT_NND_WM_FSPL',
            'outputs_dir': 'bot/tmp/results',
            'print_iter': 1,
            'save_imgs': True,
            'save_checkpoint_epochs': 5,

            # weight
            'gan_mode': 'lsgan',
            'rec_weight': 0.5,
            'CP_weight': 1.50,
            'GP_weight': 0.40,
            'cycle_weight': 0.5,
            'adv_weight': 1,
            'semantic_weight': 1,

            # training related
            'init_type': 'normal',
            'init_gain': 0.02,
            'beta1': 0.5,
            'beta2': 0.999,

            'dis_scale': 3,
            'dis_norm': 'None',
            'dis_spectral_norm': True,
            'lr_policy': 'lambda',
            'max_epoch': 1000,
            'n_epochs': 1000,
            'n_epochs_decay': 500,

            'resume': None,
            'num_residule_block': 4,
            'lr': 0.0002
        })


        self.device = get_torch_device(self.opts)
        self.G = get_generator(self.opts, self.device)
        generate_metadata(['bot/bot_test_makeup.json'], device=self.device)

        load_checkpoint(checkpoint_path, self.device, G=self.G)

        self.register_message_handler(self.handler_start, commands=['start'])
        self.register_message_handler(self.handler_photo, content_types=['photo'])


    def handler_start(self, message):
        self.reply_to(message, """Для обработки отправьте фото лица крупным планом. Оно будет растянуто до квадратного размера.""")

    def handler_photo(self, message):
        try:
            self.request_queue.put(message, block=False)
            self.reply_to(message, f'Ваш запрос {self.request_queue.qsize()} в очереди')
        except queue.Full:
            self.reply_to(message, 'Очередь запросов переполнена, попробуйте позже')
    
    def process_request(self, message):
        self.reply_to(message, f'Ваш запрос обрабатывается')
        file_info = self.get_file(message.photo[-1].file_id)
        file_name = file_info.file_path
        downloaded_file = self.download_file(file_info.file_path)
        file_name = os.path.join('bot/tmp/images/non-makeup', 'tmp_non_makeup' + os.path.splitext(file_name)[-1])
        for file in os.listdir('bot/tmp/images/non-makeup'):
            os.remove(os.path.join('bot/tmp/images/non-makeup', file))
        if not os.path.exists(self.opts.outputs_dir):
            os.mkdir(self.opts.outputs_dir)
        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        with torch.no_grad():
            try:
                generate_metadata(['bot/bot_non-makeup.json'], device=self.device)
            except:
                self.send_message(message.from_user.id, 'Лицо не распознано, попробуйте другое изображение')
                return
            dataset = MakeupDataset(self.opts, self.device, self.opts.subset_config_files, transform=False, need_pgt=False, all_comb=True, add_original_parsing=False)
            data_loader = torchdt.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=self.opts.nThreads)

            for k, data in enumerate(tqdm(data_loader)):
                non_makeup = data['non_makeup'].to(self.device)
                makeup = data['makeup'].to(self.device)
                non_makeup_parse = data['non_makeup_parse'].to(self.device)
                makeup_parse = data['makeup_parse'].to(self.device)
                z_transfer1, z_removal1 =\
                self.G.get_transfers(non_makeup, makeup, non_makeup_parse, makeup_parse)
                for j in range(z_transfer1.shape[0]):
                    i = k*data_loader.batch_size + j
                    image=(z_transfer1[j].detach() / 2 + 0.5).cpu()
                    ref = (makeup[j].detach() / 2 + 0.5).cpu()
                    torchvision.utils.save_image([image, ref], os.path.join(self.opts.outputs_dir, f'res{i}.png'))
                    self.send_photo(message.from_user.id, photo=open(os.path.join(self.opts.outputs_dir, f'res{i}.png'), 'rb'))

    def run(self):
        thread = myThread(request_queue=self.request_queue, request_handler=self.process_request)
        thread.start()
        self.infinity_polling(timeout=None, skip_pending=True, interval=1)
        self.request_queue.put(None)
        thread.join()