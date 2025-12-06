import os,time
import torch.nn.functional as F
import numpy as np 
import h5py
import torch
import torch.nn as nn
from models import get_encoder
import openslide
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from utils.file_utils import save_hdf5
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, set_start_method
from utils.constants import MODEL2CONSTANTS
import random
torch_normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)


def diff_wsi_single_feature(data, coords, noise_features, model,args):
    model.eval()
    similar='l_2'#'cos'
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    _, _, _, _, dict_data = model(data)
    A_raw = dict_data['A_raw']
    for i in range((noise_features.shape[0])):
        noise_data = noise_features[i]
        noise_data = noise_data.cuda()
        if noise_data.shape[0]>10000 and args.model_type == 'transmil':
            noise_data = noise_data[indices]
        if similar == 'l_2':
            data_normed = torch_normalizer(data)#.detach().cpu().numpy()
            noise_data_normed = torch_normalizer(noise_data)#.detach().cpu().numpy()
            diff_feature = torch.norm((data_normed - noise_data_normed), dim=1).view(1,-1)#np.linalg.norm((data_normed - noise_data_normed), axis=1).reshape(1,-1)
            #diff_feature = torch.from_numpy(diff_feature).cuda()
        else:
            diff_feature = F.cosine_similarity(data, noise_data, dim=1)#wsi_feature 为1*1024，wsi_noise_feature为92*1024
            diff_feature = diff_feature.unsqueeze(0)
            diff_feature = 1-diff_feature
            #diff_feature = diff_feature.detach().cpu().numpy()
        out = (A_raw * diff_feature).sum().detach().cpu().numpy().reshape(1, 1)
    return -out


def Stadis_val_score(net,test_loader,perturbation_feature_path,output_path,out_flag,M_list,T_list,rotation_list,diff_type,agg_type,args,oodname=''):
    if diff_type=='feature':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_diff_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_diff_val_Out_%s.txt'%(output_path,oodname)
    else:
        if out_flag == True:
            temp_file_name_val = '%s/Probability_diff_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_diff_val_Out_%s.txt'%(output_path,oodname)
    path=os.path.join(output_path,'diff')
    if not os.path.exists(path):
        os.makedirs(path)
    
    t0=time.time()
    #对数据进行多次提取
    if diff_type=='logit':
        T_list=[1]
    time_fea=0
    time_dis = 0
    t1 = time.time()
    for item_val,temperature in enumerate(T_list):
        for item_data,batch_data in enumerate(test_loader):
            if item_data%30==0:
                print('\nprogress: {}/{},speed :{}'.format(item_data,len(test_loader),time.time()-t1))
            t1 = time.time()
            #if item_data==1:
            #    continue
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]#+'.h5'
            print(id,end=' ')
            #noise_fea_path=os.path.join(perturbation_feature_path,'h5_files',id)
            if item_data%5 ==4:
                print()
            data = data.cuda()
            t_1=time.time()
            first_noise = True
            for magnitude in M_list:
                for rotation in rotation_list:
                    t_1=time.time()
                    noise_fea = make_noise_feature(perturbation_feature_path,id,args,out_flag,magnitude,rotation)
                    noise_fea = noise_fea.unsqueeze(0)#1*N*1024
                    t_2=time.time()
                    time_fea+=t_2-t_1
                    if diff_type=='softmax':
                        ...
                    elif diff_type=='feature':
                        if agg_type =='agg_noise':
                            temp_diff_Outputs=diff_wsi_feature(data, coords, noise_fea,id, net,M_list,temperature,diff_type,args)
                        elif agg_type == 'noise_agg':
                            temp_diff_Outputs=diff_wsi_single_feature(data, coords, noise_fea, net,args)
                        if first_noise:
                            score=temp_diff_Outputs
                            first_noise = False
                        else:
                            score = np.concatenate((score,temp_diff_Outputs),axis=1)
                    t_3 = time.time()
                    time_dis+=t_3-t_2

                    #perturbation_feature.append(noise_fea)
            #perturbation_feature = torch.stack(perturbation_feature)#.view(int(len(M_list)*4), -1, noise_fea.shape[1])
            
            
            #根据编号来导入ood数据。
            
            if item_data ==0:
                out=score
            else:
                out=np.concatenate((out,score),axis=0)
                #break
            
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒,计算特征时间：%.2f秒，计算StaDis时间：%.2f'%(time.time()-t0,time_fea,time_dis))    
    t0=time.time()
    g = open(temp_file_name_val, 'w')
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()

def Stadis_test_score(net,test_loader,perturbation_feature_path,output_path,out_flag,magnitude,temperature,rotation,magnitude_type,diff_type,agg_type,args,oodname=''):
    if diff_type=='feature':
        if out_flag == True:
            temp_file_name_test = '%s/Probability_diff_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_test = '%s/Probability_diff_test_Out_%s.txt'%(output_path,oodname)
    else:
        if out_flag == True:
            temp_file_name_test = '%s/Probability_diff_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_test = '%s/Probability_diff_test_Out_%s.txt'%(output_path,oodname)
    
    M_list=[magnitude]
    t0=time.time()
    if not os.path.exists(perturbation_feature_path):
        os.makedirs(perturbation_feature_path) 
    os.makedirs(os.path.join(perturbation_feature_path, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(perturbation_feature_path, 'h5_files'), exist_ok=True)
    t1=time.time()
    for item_data,batch_data in enumerate(test_loader):
        data, target, coords, id = batch_data#[21816, 1024]
        id = id[0]
        #print(id,end=' ')
        if item_data%30==0:
            print('\nprogress: {}/{},speed :{}'.format(item_data,len(test_loader),time.time()-t1))
            t1 = time.time()
        noise_fea = make_noise_feature(perturbation_feature_path,id,args,out_flag,magnitude,rotation)
        data, target = data.cuda(), target.cuda()
        if diff_type=='softmax':
            ...
        elif diff_type=='feature':
            if agg_type =='agg_noise':
                temp_diff_Outputs=test_wsi_feature(data, coords, noise_fea, net,args)
            elif agg_type == 'noise_agg':
                temp_diff_Outputs=diff_wsi_single_feature(data, coords, noise_fea, net,args)
        if item_data ==0:
            out=temp_diff_Outputs
        
        else:
            out=np.concatenate((out,temp_diff_Outputs),axis=0)
            #break
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time() 
    f = open(temp_file_name_test, 'w')
    for i in range(out.shape[0]):
        f.write("{}\n".format(out[i,0]))
    f.close()

def rand_bbox(inputs_size, dst_size):
    random.seed(42)
    x = random.randint(0, inputs_size-dst_size)
    y = random.randint(0, inputs_size-dst_size)
    return x, y, x+dst_size, y+dst_size

def Cut_Thumbnail(args,out_flag,wsi,coords,img_transforms,thu_path):
	if args.experiment_mask == 'C16' and out_flag:
		level = 4
	else:
		level = wsi.level_count - 1
	downsample_factor = wsi.level_downsamples[level]
	if args.model_name == 'conch_v1':
		size = 448
	else:
		size = 224
	#x = torch.ones(coords.shape[0])
	#y = torch.ones(coords.shape[0])
	all_slide_data = []   
	dest_files = os.listdir(thu_path)
	for i in range(coords.shape[0]):
		name ='%s_%s.pt'%(str(coords[i][0]), str(coords[i][1]))
		noise_fea_path=os.path.join(thu_path,name)
		if  name in dest_files:
			slide_data = torch.load(noise_fea_path)
		else:
			if coords[i][0]+size/2-(size/2)*downsample_factor>0:
				x_start = int(coords[i][0]+size/2-(size/2)*downsample_factor)
			else:
				x_start = 0        
			if coords[i][1]+size/2-(size/2)*downsample_factor>0:
				y_start = int(coords[i][1]+size/2-(size/2)*downsample_factor)
			else:
				y_start = 0                           
			if size>wsi.level_dimensions[0][0]-coords[i][0]:
				x = wsi.level_dimensions[0][0]-coords[i][0]
			else:
				x = size
			if size > wsi.level_dimensions[0][1]-coords[i][1]:
				y = wsi.level_dimensions[0][1]-coords[i][1]
			else:
				y = size

			slide_data=wsi.read_region((x_start,y_start), level, (x,y)).convert('RGB') 
			slide_data = img_transforms(slide_data)
			if slide_data.shape[1]!=size:
				slide_data = slide_data[:,:size,:]
			elif slide_data.shape[2]!=size:
				slide_data = slide_data[:,:,:size]
			torch.save(slide_data, noise_fea_path)
		all_slide_data.append(slide_data)
	slide_data = torch.stack(all_slide_data)
	return slide_data

def compute_single_w_loader(out_flag,img_transforms, loader, model,args, verbose = 0,magnitude=10,rotation=-1,wsi = '', thu_path =''):
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))
	mode = 'w'
	features_list = []
	if rotation < 37 and rotation > 32:
		slide_data=wsi.read_region((0,0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1]).convert('RGB') 
		slide_data = img_transforms(slide_data)

	for count, data in enumerate((loader)):
		batch = data['img'].cuda()
		coords = data['coord'].numpy().astype(np.int32)

		with torch.inference_mode():	
			if rotation < 41 and rotation > 36:
				slide_data = Cut_Thumbnail(args,out_flag,wsi,coords,img_transforms,thu_path )
				bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(2), int(batch.size(2)/4))
				batch[:,:, bbx1:bbx2, bby1:bby2] = F.interpolate(slide_data, (int(batch.size(2)/4), int(batch.size(2)/4)))
			if rotation<60:
				
				gradient = torch.ones_like(batch)
				gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) * (0.229/0.229))
				gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) * (0.224/0.229))
				gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) * (0.225/0.229)) 
				temp_x_noise=torch.add(batch.data, gradient,alpha= -magnitude)
			
                
			if rotation%4 == 1:
				temp_x_noise = torch.rot90(temp_x_noise, 1, [2, 3])
			elif rotation%4 ==2:
				temp_x_noise = torch.rot90(temp_x_noise, 2, [2, 3])
			elif rotation%4 == 3:
				temp_x_noise = torch.rot90(temp_x_noise, 3, [2, 3])
			else:
				temp_x_noise=temp_x_noise
			#if rotation < 41 or rotation >44:
			#	index, lam = prepare_mixup(temp_x_noise, 1)
			#	temp_x_noise = mixing(temp_x_noise, index, lam)
			if rotation <9 and rotation>4:#增加了cut-thumbnail
				bbx1, bby1, bbx2, bby2 = rand_bbox(temp_x_noise.size(2), int(temp_x_noise.size(2)/4))
				temp_x_noise[:,:, bbx1:bbx2, bby1:bby2] = F.interpolate(temp_x_noise[:, :, :, :], (int(temp_x_noise.size(2)/4), int(temp_x_noise.size(2)/4)))
			elif rotation < 33 and rotation > 28:
				bbx1, bby1, bbx2, bby2 = rand_bbox(temp_x_noise.size(2), int(temp_x_noise.size(2)/4))
				temp_x_noise[:,:, bbx1:bbx2, bby1:bby2] = F.interpolate(batch.data, (int(temp_x_noise.size(2)/4), int(temp_x_noise.size(2)/4)))
			elif rotation < 37 and rotation > 32:
				if slide_data.dim() == 2:  # 如果是 (H, W)
					slide_data = slide_data.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, H, W)
				elif slide_data.dim() == 3:  # 如果是 (N, H, W)
					slide_data = slide_data.unsqueeze(0)
				bbx1, bby1, bbx2, bby2 = rand_bbox(temp_x_noise.size(2), int(temp_x_noise.size(2)/4))
				temp_x_noise[:,:, bbx1:bbx2, bby1:bby2] = F.interpolate(slide_data, (int(temp_x_noise.size(2)/4), int(temp_x_noise.size(2)/4)))
            
			if rotation < 53 and rotation > 40:
				slide_data = Cut_Thumbnail(args,out_flag,wsi,coords,img_transforms,thu_path )
				bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(2), int(batch.size(2)/4))
				batch[:,:, bbx1:bbx2, bby1:bby2] = F.interpolate(slide_data, (int(batch.size(2)/4), int(batch.size(2)/4)))
			if rotation >48:
				index, lam = prepare_mixup(temp_x_noise, 1)
				temp_x_noise = mixing(temp_x_noise, index, lam)
			if rotation<13 and rotation>8:
				constants = MODEL2CONSTANTS[args.model_name]
				mean = constants['mean']
				std = constants['std']
				green_normalized = [(0 - mean[i]) / std[i] if i != 1 else (1 - mean[i]) / std[i] for i in range(3)]#绿色
				blue_normalized = [(0 - mean[i]) / std[i] if i != 2 else (1 - mean[i]) / std[i] for i in range(3)]#绿色
				red_normalized = [(0 - mean[i]) / std[i] if i != 0 else (1 - mean[i]) / std[i] for i in range(3)]#绿色
				for i in range(2):
					for j in range(2):
						temp_x_noise[:,0, int(temp_x_noise.shape[2]/2) + i, int(temp_x_noise.shape[3]/2) + j] = green_normalized[0]  # R通道
						temp_x_noise[:,1, int(temp_x_noise.shape[2]/2) + i, int(temp_x_noise.shape[3]/2) + j] = green_normalized[1]  # G通道
						temp_x_noise[:,2, int(temp_x_noise.shape[2]/2) + i, int(temp_x_noise.shape[3]/2) + j] = green_normalized[2]
				for i in range(2):
					for j in range(2):
						temp_x_noise[:,0, int(temp_x_noise.shape[2]/4) + i, int(temp_x_noise.shape[3]/4) + j] = blue_normalized[0]  # R通道
						temp_x_noise[:,1, int(temp_x_noise.shape[2]/4) + i, int(temp_x_noise.shape[3]/4) + j] = blue_normalized[1]  # G通道
						temp_x_noise[:,2, int(temp_x_noise.shape[2]/4) + i, int(temp_x_noise.shape[3]/4) + j] = blue_normalized[2]
				for i in range(2):
					for j in range(2):
						temp_x_noise[:,0, int(3*temp_x_noise.shape[2]/4) + i, int(3*temp_x_noise.shape[3]/4) + j] = red_normalized[0]  # R通道
						temp_x_noise[:,1, int(3*temp_x_noise.shape[2]/4) + i, int(3*temp_x_noise.shape[3]/4) + j] = red_normalized[1]  # G通道
						temp_x_noise[:,2, int(3*temp_x_noise.shape[2]/4) + i, int(3*temp_x_noise.shape[3]/4) + j] = red_normalized[2]                
			elif rotation<17 and rotation>12:
				np.random.seed(0)
				constants = MODEL2CONSTANTS[args.model_name]
				mean = constants['mean']
				std = constants['std']
				patches = []
				mask_color = [(0 - mean[i]) / std[i] for i in range(3)]
				for i in range(0, temp_x_noise.shape[2], 16):
					for j in range(0, temp_x_noise.shape[3], 16):
						patches.append((i, j))
				mask_indices = np.random.choice(len(patches), int(len(patches)/3), replace=False)
				for idx in mask_indices:
					i, j = patches[idx]
					temp_x_noise[:,0, i:i+16, j:j+16] = mask_color[0]
					temp_x_noise[:,1, i:i+16, j:j+16] = mask_color[1]
					temp_x_noise[:,2, i:i+16, j:j+16] = mask_color[2]                        
			elif rotation<25 and rotation>20:
				np.random.seed(0)
				constants = MODEL2CONSTANTS[args.model_name]
				mean = constants['mean']
				std = constants['std']
				patches = []
				#=mask_color = [(0 - mean[i]) / std[i] if i != 0 else (1 - mean[i]) / std[i] for i in range(3)]#红色
				mask_color = [(0 - mean[i]) / std[i] for i in range(3)]
				for i in range(0, temp_x_noise.shape[2], 16):
					for j in range(0, temp_x_noise.shape[3], 16):
						patches.append((i, j))
				mask_indices = np.random.choice(len(patches), int(4*len(patches)/5), replace=False)
				for idx in mask_indices:
					i, j = patches[idx]
					temp_x_noise[:,0, i:i+16, j:j+16] = mask_color[0]
					temp_x_noise[:,1, i:i+16, j:j+16] = mask_color[1]
					temp_x_noise[:,2, i:i+16, j:j+16] = mask_color[2]			
			elif rotation<29 and rotation>24:
				np.random.seed(0)
				constants = MODEL2CONSTANTS[args.model_name]
				mean = constants['mean']
				std = constants['std']
				patches = []
				#=mask_color = [(0 - mean[i]) / std[i] if i != 0 else (1 - mean[i]) / std[i] for i in range(3)]#红色
				mask_color = [(0 - mean[i]) / std[i] if i != 0 else (1 - mean[i]) / std[i] for i in range(3)]
				for i in range(0, temp_x_noise.shape[2]):
					for j in range(0, temp_x_noise.shape[3]):
						patches.append((i, j))
				mask_indices = np.random.choice(len(patches), int(1*len(patches)/5), replace=False)
				for idx in mask_indices:
					i, j = patches[idx]
					temp_x_noise[:,0, i:i+16, j:j+16] = mask_color[0]
					temp_x_noise[:,1, i:i+16, j:j+16] = mask_color[1]
					temp_x_noise[:,2, i:i+16, j:j+16] = mask_color[2]	            
				
				
				
			#temp_x_noise = temp_x_noise.to(device, non_blocking=True)
			if args.model_name == 'conch_v1':
				features = model.encode_image(temp_x_noise, proj_contrast=False, normalize=False)
			else:
				features = model(temp_x_noise)
			#features = model(temp_x_noise)
			features_list.append(features)
			#features = features.cpu().numpy().astype(np.float32)
            
			#asset_dict = {'features': features, 'coords': coords}
			#save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	fea = torch.cat(features_list, dim=0)
	return fea
def prepare_mixup(batch, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda."""
    np.random.seed(0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    return index, lam


def mixing(data, index, lam):
    return lam * data + (1 - lam) * data[index]

def make_noise_feature(feature_path,id,args,out_flag,magnitude,rotation):
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    loader_kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}
    bag_name = id+'_%s_%s'%(str(magnitude),str(rotation))+'.h5'
    pt_name=id+'_%s_%s'%(str(magnitude),str(rotation))
    if out_flag:#ID
        h5_file_path = os.path.join(args.id_data_h5_dir, 'patches', id+'.h5')#从之前的pt中获取corrds
        output_path = os.path.join(feature_path, 'h5_files', bag_name)#存放扰动特征的地方
        csv_path = args.id_csv_path
    else:
        if args.ood_dataset_name == 'RCC_Normal':
            args.ood_data_h5_dir = os.path.join(args.data_h5_dir, 'RCC_ood') 
            h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', id+'.h5')
            output_path = os.path.join(feature_path, 'h5_files', bag_name)
            csv_path = os.path.join(args.csv_path,'RCC_ood','process_list_autogen.csv')
            if not os.path.exists(h5_file_path):
                args.ood_data_h5_dir = os.path.join(args.data_h5_dir, args.experiment_mask) 
                h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', id+'.h5')
                csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
        else:
            args.ood_data_h5_dir = os.path.join(args.data_h5_dir, args.ood_dataset_name) 
            h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', id+'.h5')
            output_path = os.path.join(feature_path, 'h5_files', bag_name)
            args.ood_csv_path = os.path.join(args.csv_path,args.ood_dataset_name,'process_list_autogen.csv')
            csv_path = args.ood_csv_path
    pt_path = os.path.join(feature_path, 'pt_files')
    os.makedirs(pt_path, exist_ok=True)
    os.makedirs(os.path.join(feature_path,'h5_files'), exist_ok=True)
    dest_files = os.listdir(pt_path)
    if pt_name+'.pt' in dest_files:
        #print('skipped {}'.format(pt_name))
        #noise_fea_path=os.path.join(feature_path,'h5_files',id+'_%s_%s'%(str(magnitude),str(rotation))+'.h5')
        #with h5py.File(noise_fea_path,'r') as hdf5_file:
        #    noise_features = hdf5_file['features'][:]
        #从pt中读取文件
        tt=time.time()
        noise_fea_path=os.path.join(feature_path,'pt_files',pt_name+'.pt')
        noise_features = torch.load(noise_fea_path, map_location=torch.device('cpu'))
        #tt1=time.time()
        #noise_fea_path=os.path.join(feature_path,'h5_files',id+'_%s_%s'%(str(magnitude),str(rotation))+'.h5')
       # with h5py.File(noise_fea_path,'r') as hdf5_file:
        #    noise_features_h5 = hdf5_file['features'][:]
        #tt2 = time.time()
        #print('load时间：%.2f,h5时间：%.2f'%(tt1-tt,tt2-tt1))
        return noise_features
    model = model.to(device)
    _ = model.eval()
    df = pd.read_csv(csv_path)
    slide_file_path = df[df['slide_id'].str.contains(id)]['slide_id'].iloc[0]
    #35服务器操作
    time_start = time.time()
    if 'data_nas/' in slide_file_path:
        slide_file_path=slide_file_path.replace('data_nas/', 'data_nas1/')
    wsi = openslide.open_slide(slide_file_path)
    dataset_patch = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

    loader = DataLoader(dataset=dataset_patch, batch_size=args.batch_size, **loader_kwargs)
    thu_path = os.path.join(feature_path,'thu_data',id)
    os.makedirs(thu_path, exist_ok=True)
    features = compute_single_w_loader(out_flag,img_transforms, loader = loader, model = model, args = args, \
									   verbose = 0,magnitude=magnitude,rotation=rotation,wsi = wsi,thu_path =thu_path)
    torch.save(features, os.path.join(feature_path, 'pt_files', pt_name+'.pt'), pickle_protocol=4)

    return features

def test_wsi_feature(data, coords, noise_fea, model,args):
    model.eval()
    similar = 'l_2'
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    _, _, _, wsi_feature, result_dict = model(data)
    A = result_dict['A_raw']
    wsi_feature = torch.mm(A,data)
    wsi_feature = wsi_feature.detach().cpu()
    data = data.detach().cpu()
    coords = np.array(coords) 
    #noise_features = noise_fea.cuda()
    
    #_, _, _, wsi_noise_feature, _ = model(noise_features)
    if noise_fea.shape[0]>10000 and args.model_type == 'transmil':
        noise_fea = noise_fea[indices]
    wsi_noise_feature = torch.mm(A,noise_fea.cuda())
    wsi_noise_feature = wsi_noise_feature.detach().cpu()
    # wsi_noise_feature = torch.matmul(A_raw, noise_features)
    if similar == 'l_2':
        wsi_feature_normed = normalizer(wsi_feature.numpy())
        wsi_noise_feature_normed = normalizer(wsi_noise_feature.numpy())
        diff_feature = np.linalg.norm(wsi_feature_normed - wsi_noise_feature_normed)
        out = np.array([[diff_feature]])
    else:
        diff_feature = F.cosine_similarity(wsi_feature, wsi_noise_feature, dim=1)#wsi_feature 为1*1024，wsi_noise_feature为92*1024
        diff_feature = 1-diff_feature
        out=diff_feature.detach().cpu().numpy().reshape(diff_feature.shape[0],1)
    return -out

def process_feature(args):
    item_mag, rotation, A_raw, wsi_feature, noise_features, coords, noise_coords, order = args
     
    if not order:
        sorted_features = np.zeros_like(noise_features)
        for i, coord in enumerate(coords):
            j = np.where((noise_coords == coord).all(axis=1))[0][0]
            sorted_features[i] = noise_features[j]
        noise_features = sorted_features
    noise_features = torch.from_numpy(noise_features).cuda().detach()
    wsi_noise_feature = torch.matmul(A_raw, noise_features)
    diff_feature = F.cosine_similarity(wsi_feature, wsi_noise_feature, dim=1)
    diff_feature = 1 - diff_feature
    return diff_feature.detach().cpu().numpy().reshape(diff_feature.shape[0], 1),item_mag,rotation

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def diff_wsi_feature(data, coords, noise_features,id, model,M_list,temperature,diff_type,args):
    model.eval()
    similar='l_2'#'cos'
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    _, _, _, _, result_dict = model(data)
    A = result_dict['A_raw']
    wsi_feature = torch.mm(A,data)
    coords = np.array(coords)
    wsi_feature = wsi_feature.detach().cpu()
    wsi_noise_feature = torch.zeros(noise_features.shape[0],wsi_feature.shape[1])
    with torch.no_grad():
        for i in range((noise_features.shape[0])):
            noise_data = noise_features[i]
            noise_data = noise_data.cuda()
            #_, _, _, noise_fea, _ = model(noise_data)
            if noise_data.shape[0]>10000 and args.model_type == 'transmil':
                noise_data = noise_data[indices]
            noise_fea = torch.mm(A,noise_data)
            wsi_noise_feature[i]=noise_fea.detach().cpu().squeeze()
        # wsi_noise_feature = torch.matmul(A_raw, noise_features)
    #noise_features = noise_features.cuda()
    #A_raw_expanded = A_raw.expand(noise_features.shape[0], 1, -1)
    #wsi_noise_feature = torch.matmul(A_raw_expanded, noise_features)
    #wsi_noise_feature = wsi_noise_feature.squeeze(1)#(92, 1024)
    if similar == 'l_2':
        wsi_feature_normed = normalizer(wsi_feature.numpy())
        wsi_noise_feature_normed = normalizer(wsi_noise_feature.numpy())
        diff_feature = np.linalg.norm(wsi_feature_normed - wsi_noise_feature_normed)
        out = np.array([[diff_feature]])
    else:
        diff_feature = F.cosine_similarity(wsi_feature, wsi_noise_feature, dim=1)#wsi_feature 为1*1024，wsi_noise_feature为92*1024
        diff_feature = diff_feature.unsqueeze(0)
        diff_feature = 1-diff_feature
        out=diff_feature.detach().cpu().numpy()
    return -out

