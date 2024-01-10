import sys

sys.path.append('.')
import json
import logging
import math
import os
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.transforms as trns
import tqdm
from vit_pytorch import ViT
from ResNetModified import ResNetModified
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from transformers import optimization
from val_test import validate

# from data_load.magnet_dataset import magnet_Dataset
from data_load.new_magnet_dataset import magnet_Dataset
logging.basicConfig(level=logging.DEBUG)

import os

def generate_sequence(T_0, T_mult, n):
    sequence = []
    term = T_0
    for i in range(n):
        term = (T_0*(1-T_mult**(i+1)))   /    (1-T_mult)
        sequence.append(term)

    return sequence


def remove_directory(directory):
    dir_weight = os.path.join(dir_save, 'weight_cbam2')
    os.makedirs(dir_weight, exist_ok=True)
    print(f"Saving Weight to :{dir_weight}")
    dir_log1 = os.path.join(dir_save, 'log_cbam2')
    return dir_weight,dir_log1



def factor_process(factor):
    temp  = torch.tensor([[factor[0][0],factor[1][0],factor[2][0],factor[3][0],factor[4][0]]],dtype=torch.float32)
    
    for i in range(1,len(factor[0])):
           temp2  = torch.tensor([[factor[0][i],factor[1][i],factor[2][i],factor[3][i],factor[4][i]]],dtype=torch.float32)
           temp = torch.cat([temp,temp2],dim=0)
    return temp


def main():

    #save model weight and trainning information
    current_step = 0
    _,dir_log1 = remove_directory(dir_save)
    train_writer = SummaryWriter(dir_log1)
    
    # data_loading 
    train_transform = trns.Compose([
        trns.Resize((image_size, image_size)),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create train/valid datasets
    train_set = magnet_Dataset(root=dir_dataset,frequency = frequency, 
                        transform=train_transform ,data_num = data_num ,Mtype = "Train",data_std=data_std)
    val_set = magnet_Dataset(root=dir_dataset,frequency = frequency, 
                        transform=train_transform ,data_num = data_num ,Mtype = "Val")
    
    if data_std == True :
        Z_score = {
            'mean' : train_set.mean,
            'std' : train_set.std
        }
        
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2,prefetch_factor = 1,pin_memory=True)
    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=2,prefetch_factor = 1,pin_memory=True)
    
    single_epoch = math.ceil(len(train_set)/batch_size)
    max_step = single_epoch*TOTAL_EPOCHS


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2_Regularization,eps=1e-7)
    lr_schedulers = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=(8*single_epoch),T_mult=2,eta_min=lr_end)

    model.train()
    loss = OrderedDict()

    hyperparameters = {
        "Epoch": TOTAL_EPOCHS,
        "Step" : max_step,
        "Batch Size"  : batch_size,
        "Init Learning Rate" : lr,
        "End Learning Rate" : lr_end,
        'optimizer': type(optimizer).__name__,
        "Optimizer defaults" : (optimizer.defaults),
        'epoch_cycle:' : (epoch_cycle),
        "註解" : tips,
    }
    
    
    with open(os.path.join(dir_save,"parameter.txt"), 'w',encoding='utf-8') as file:
        file.write(json.dumps(hyperparameters, ensure_ascii=False, indent=4))
    scaler = GradScaler()
    for epoch in range(TOTAL_EPOCHS):
        tqdm_loader = tqdm.tqdm(train_loader)
        
        if epoch == 1:
            start = time.time()
        for images, targets,factor,_ in tqdm_loader:
            
            current_step += 1
            images = images.cuda()
            # targets = targets.cuda()
            
            target=[]
            for i in range(len(targets)):
                target.append([targets[i]])
            
            del targets
            
            target = torch.FloatTensor(target).cuda()
            with autocast():
                # x1,x2,x3,x4 = model(images)
                # logging.info(x1.size())
                # logging.info(x2.size())
                # logging.info(x3.size())
                # logging.info(x4.size())
                
                out_cls = model(images)
                losses = criterion(out_cls,target)
                

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            
            lrate = optimizer.state_dict()['param_groups'][0]['lr']
            train_writer.add_scalar("loss", losses.item(), global_step=current_step)
            train_writer.add_scalar("lr", lrate, global_step=current_step)
            train_writer.flush()
            loss['loss'] = losses.item()
            loss['lr'] = lrate
            tqdm_loader.set_postfix(loss)
            lr_schedulers.step()
            tqdm_loader.set_description(f'epoch:{epoch+1}'.ljust(11)  + f'step:<{current_step}/{max_step}>' .ljust(21))       

       
        if  (epoch+1) in epoch_cycle:
            logging.info("Training data Acc")
            validate(model, epoch+1, train_loader, dir_save,data_std = data_std,Training = True,value_recover = Z_score)
            logging.info("Validation data Acc")
            validate(model, epoch+1, val_loader, dir_save,data_std = data_std,Training = False,value_recover = Z_score)
            
        model.train()
        if epoch == 1:
            end = time.time()
            cost_time = (end-start)*TOTAL_EPOCHS
            now = time.time()
            end_time = time.ctime((now+cost_time))
            logging.info("Estimated end time : " + end_time)
            

if __name__ == '__main__':

    torch.manual_seed(0)
    device_ids = [0]
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.benchmark = True

   
    
    epoch_cycle = [8, 24, 56, 120, 240, 500, 1000, 2000,4000,4088]
    lr_end = 1e-10
    lr = 0.001
    TOTAL_EPOCHS = 4089
    batch_size = 32
    image_size = 256
    save_interval = 500
    L2_Regularization = 0.2
    CNN_feature = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    VIT_feature = model = ViT(
        image_size = 64,
        patch_size = 16,
        channels=512,
        num_classes = 1,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    model = ResNetModified(CNN_feature,VIT_feature)
    data_num = 0
    frequency = 200
    ####################################
    criterion = nn.MSELoss()
    tips = (f"mobile Vit 優化器Rmsprop,",
            f"image_size=256",
            f"epoch更改為{TOTAL_EPOCHS}",
            f"lr 上下限修正，修改學習率策略為CosineAnnealingWarmRestarts，每{epoch_cycle}Epoch回復最高點，固定epoch為一周",
            f"使用MSE，修該laynormal改為batchnormal",
        )
    
    
    data_std = False
    ####################################
    dir_dataset = r'laser_image/crop_image/ring/'
    dir_save_path = r'output/ring/version12/version0_'
    dir_save = os.path.join(dir_save_path+str(frequency)+"hz",'0'+str(data_num+1))
    main()