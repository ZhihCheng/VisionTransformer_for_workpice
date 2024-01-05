import os
import glob
import math
import pandas as pd
class Reader_Data:


    def get_txt_data(txt_path):
        f = open(txt_path+'magnet_data.txt')
        temp = f.readline().strip('\n').split(' ')
        f.close()
        return temp

    def get_total_data(path, data_num, frequency,model_type, set_num=-1):
        work_pice_name = []
        img = []
        target = []
        factor = []
        out_side = []
        out_name = []
        for i in range(1,4):
            excel_file_name = os.path.join(path,'batch'+str(i)+'_ring_data.xlsx')
            print(excel_file_name)
            sheet_name = 'factor'
            factor_matrix = pd.read_excel(excel_file_name, sheet_name=sheet_name,header=None)
            sheet_name = str(frequency) + 'hz_' + ('pmb' if data_num ==0 else 'iron')
            data_matrix = pd.read_excel(excel_file_name, sheet_name=sheet_name,header=None)
            image_path = os.path.join(path,f"{model_type}_1024")

            for c in range(1,10):
                for k in (range(1,5) if model_type=='Train' else range(5,7)):
                    set_image_dir = f'trail {i}_{c}_0{k}'
                    if data_matrix[k-1][c-1] != 'x':
                        for t in glob.glob(os.path.join(image_path,set_image_dir,'*.jpg')):
                                work_pice_name.append(f'trail {i}_{c}_0{k}')
                                img.append(t)
                                target.append(data_matrix[k-1][c-1])
                                factor.append(factor_matrix.iloc[c-1].to_numpy())
                    else:
                        out_side.append(f'trail {i}_{c}_0{k}')

        return img, target, factor, work_pice_name, out_side
    
    def get_TEST_data(path):
        img = []
        target = []
        path = os.path.join(path,"Test_1024")
        for j in os.listdir(path):
            for k in glob.glob(path+'/'+j+'/*.jpg'):
                img.append(k)
        return img, target