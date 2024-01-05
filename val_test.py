import torch
import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from torch.cuda.amp import autocast
import os
def factor_process(factor):
    temp  = torch.tensor([[factor[0][0],factor[1][0],factor[2][0],factor[3][0],factor[4][0]]],dtype=torch.float32)
    
    for i in range(1,len(factor[0])):
           temp2  = torch.tensor([[factor[0][i],factor[1][i],factor[2][i],factor[3][i],factor[4][i]]],dtype=torch.float32)
           temp = torch.cat([temp,temp2],dim=0)
    return temp


def validate(model, epoch, validation_data, save_path,data_std = False,Training = False,value_recover = None):
    # 設置模型為評估模式
    model.eval()
    y_val = []
    y_pred = []
    tqdm_loader = tqdm.tqdm(validation_data)
    with torch.no_grad():
        for images, targets,factor,_ in tqdm_loader:
            target=[]
            images = images.cuda()

            for i in range(len(targets)):
                target.append([targets[i]])

            # factors = factor_process(factor)
            # factors = factors.cuda()
            target = torch.FloatTensor(target).cuda()
            with autocast():
                out_cls = model(images)
            y_pred.extend(out_cls.cpu().numpy())
            y_val.extend(target.cpu().numpy())
            
        # 還原數據
        # recovered_target = normalized_target * std + mean
        print(value_recover['std'])
        print(value_recover['mean'])
        if data_std == True:
            y_pred = np.array(y_pred)
            y_pred = y_pred * value_recover['std'] + value_recover['mean']
            if Training:
                y_val = np.array(y_val)
                y_val = y_val * value_recover['std'] + value_recover['mean']

            
        y_pred = np.nan_to_num(y_pred)  # 將 NaN 和無窮大數值替換為 0
        # 計算 R2 分數
        r2 = r2_score(y_val, y_pred)
        # 計算 MSE 分數
        mse = mean_squared_error(y_val, y_pred)
        # 計算 MAE 分數
        mae = mean_absolute_error(y_val, y_pred)
        # 輸出分數和 epoch
        print(f"Epoch: {epoch}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MSE Score: {mse:.4f}")
        print(f"MAE Score: {mae:.4f}")
        # 儲存分數和 epoch 到檔案
        with open(os.path.join(save_path,'result.txt'), 'a') as file:
            if Training :
                file.write(f"==========Train==========\n")
            else:
                file.write(f"==========Validation==========\n")
            file.write(f"Epoch: {epoch}\n")
            file.write(f"R2 Score: {r2:.4f}\n")
            file.write(f"MSE Score: {mse:.4f}\n")
            file.write(f"MAE Score: {mae:.4f}\n")
            file.write("\n")
        save_path = os.path.join(save_path,'Epoch_%d.pth' % epoch)
        state_dict = model.state_dict() 
        torch.save(state_dict, save_path)
        # except:
        #     print(np.isnan(y_pred).any())   # 檢查是否有 NaN
        #     print(np.isinf(y_pred).any())   # 檢查是否有無窮大)
        
        
        