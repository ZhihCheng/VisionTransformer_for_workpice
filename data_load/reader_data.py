import os
import glob
import math
from PIL import Image
class Reader_Data:


    def get_txt_data(txt_path):
        f = open(txt_path+'magnet_data.txt')
        temp = f.readline().strip('\n').split(' ')
        f.close()
        return temp

    def get_total_data(path, data_num, frequency,model_type, set_num=-1):
        img = []
        target = []
        factor = []
        out_side = []
        out_name = []
        if frequency == 50:
            target_path =  os.path.join(path,"value_data_50hz")           
        elif frequency == 200:
            target_path =  os.path.join(path,"value_data_200hz")  
        elif frequency == 400:
            target_path =  os.path.join(path,"value_data_400hz")  
        elif frequency == 800:
            target_path =  os.path.join(path,"value_data_800hz")
        elif frequency == 'bone':
            target_path =  os.path.join(path,"value_data")


        if model_type == "Train" :
            image_path = os.path.join(path,"Train_1024")
            target_path = os.path.join(target_path,"Train")
        elif model_type == "Val" :
            image_path = os.path.join(path,"Val_1024")
            target_path = os.path.join(target_path,"Test")
        else:
            image_path = os.path.join(path,"Test_1024",str(set_num))
            target_path = os.path.join(target_path,"Test")
        



        print("Image Path from : " + image_path)
        print("Target Path From : " + target_path)
        for c,j in enumerate(os.listdir(image_path)):
            f = open(os.path.join(target_path,j,'magnet_data.txt'))
            target_temp = f.readline().strip('\n').split(' ')
            f.close()
            target_temp =  list(map(float, target_temp))
            if target_temp[data_num]== 0:
                out_name.append(j)
                if len(out_side) == 0 :
                    out_side.append(c)
                elif out_side[len(out_side)-1] != j :
                    out_side.append(c)
                continue
            for k in glob.glob(os.path.join(image_path,j,'*.jpg')):
                #target########
                
                target.append(target_temp[data_num])
                ###############
                img.append(k)

                #Factor########
                f = open(os.path.join(image_path,j,'factor.txt'))
                temp= f.readline().strip('\n').split(' ')
                f.close()
                temp =  list(map(float, temp))
                factor.append(temp)
                ###############
        return img, target, factor, out_side,out_name
    
    def get_TEST_data(path):
        img = []
        target = []
        path = os.path.join(path,"Test_1024")
        for j in os.listdir(path):
            for k in glob.glob(path+'/'+j+'/*.jpg'):
                img.append(k)
        return img, target