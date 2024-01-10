import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_load.new_reader_data import Reader_Data
import pandas as pd
class magnet_Dataset(Dataset):
    def __init__(self, root, frequency, Mtype, transform=None, data_num='Null',  set_num=1,data_std=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform
        self.imgs = []
        self.lbls = []
        self.factor = []
        self.img_path = []
        self.work_pice_name = []
        self.set_num = set_num
        self.data_num = data_num
        self.frequency = frequency
        self.model_type = Mtype
        self.out_side = []
        self.imgs ,self.lbls ,self.factor ,self.work_pice_name, self.out_side = Reader_Data.get_total_data(root, self.data_num, self.frequency, self.model_type, self.set_num)

        # print(f'imags len = {len(self.imgs)}')
        # print(f'traget len = {len(self.lbls)}')
        # print(f'factor len = {len(self.factor)}')
        # print(f'work_pice_name len = {len(self.work_pice_name)}')
        print(f'outlier data= {self.out_side}')
        self.avglbl = np.mean(self.lbls)
        if data_std == True:
            if self.model_type == 'Train':
                print("Data Standardization")
                # 計算平均值和標準差
                self.mean = np.mean(self.lbls)
                self.std = np.std(self.lbls)
                # 標準化數據
                self.lbls = (self.lbls - self.mean) / self.std
            
 
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        assert 0 not in self.lbls, "The list contains 0."
        # if out_name != []:
        #     for n in out_name:
        #         print(f"Remove Data : {n}")

                
        # print("total_len = ",len(self.lbls))


    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        path = imgpath[-22:]
        lbl = self.lbls[index]
        factor = self.factor[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl,factor,path
    


    def __len__(self):
        # --------------------------------------
        # ------
        # Indicate the total size of the dataset
        # -------------------
        # -------------------------
        return len(self.imgs)


