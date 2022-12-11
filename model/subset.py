from collections import namedtuple

class SubsetConfig(object):
    def __init__(self, data_type, lms_dir, mask_dir, image_dir, list_mode, filename_list):
        self.data_type = data_type
        self.lms_dir = lms_dir
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.list_mode = list_mode
        self.filename_list = filename_list


DataItem = namedtuple('DataItem', ['image_file_name', 'subset_config'])