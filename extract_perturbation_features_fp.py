import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP,Dataset_All_Bags_selection
from models import get_encoder
from ood_utils.utils_Stadis import ood_dataset_splits
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def prepare_mixup(batch, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda."""
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
def compute_w_loader(output_path, loader, model,args, verbose = 0,m_list=[]):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))
	model = model.eval()
	mode = 'w'
	for count, data in enumerate((loader)):
		with torch.inference_mode():	
			batch = data['img'].cuda()
			coords = data['coord'].numpy().astype(np.int32)
			for item_mag, magnitude in enumerate(m_list):
				for rotation in range(4):
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
					index, lam = prepare_mixup(temp_x_noise, 1)
					temp_x_noise = mixing(temp_x_noise, index, lam)
					#temp_x_noise = temp_x_noise.to(device, non_blocking=True)
					if args.model_name == 'conch_v1':
						features = model.encode_image(temp_x_noise, proj_contrast=False, normalize=False)
					else:
						features = model(temp_x_noise)
					features = features.cpu().numpy().astype(np.float32)
					if item_mag==0 and rotation == 0:
						asset_dict = {'features_%d_%d'%(item_mag,rotation): features, 'coords': coords}
					else:
						asset_dict = {'features_%d_%d'%(item_mag,rotation): features}#, 'coords': coords
					save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
					mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
#需要修改的参数
parser.add_argument('--experiment_mask', type = str,default='C16_conch',choices=['RCC','C16','RCC_conch','C16_conch'],
					help='name about experiment')
parser.add_argument('--model_name', type=str, default='conch_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'resnet50'])
parser.add_argument('--task', type=str, default='task_1_tumor_vs_normal', choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])




#default
parser.add_argument('--feat_dir', type=str, default='./')
parser.add_argument('--ood_feat_dir', type=str, default='./')#需要修改
# parser.add_argument('--data_root_dir', type=str, default='./clam_format_feature/', 
#                    help='data directory')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--dataset_csv', type=str, default='./dataset_csv/')
parser.add_argument('--data_h5_dir', type=str, default='./clam_format_patch/')
parser.add_argument('--csv_path', type=str, default='./clam_format_patch/')
parser.add_argument('--slide_id_csv', type=str, default='./dataset_csv/')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--slide_ext', type=str, default= '.svs')

args = parser.parse_args()
args.feat_dir = os.path.join(args.feat_dir,'clam_format_feature_%s'%(args.model_name))
args.ood_feat_dir_temp = os.path.join(args.ood_feat_dir,'clam_format_feature_%s'%(args.model_name))
args.id_data_h5_dir = os.path.join(args.data_h5_dir,args.experiment_mask)
args.id_csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
args.id_slide_id_csv = os.path.join(args.slide_id_csv,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
args.feat_dir = os.path.join(args.feat_dir,'%s_perturbation'%(args.experiment_mask))
if __name__ == '__main__':
	print('initializing dataset')
	if args.experiment_mask in ['C16','C16_conch']:
		args.slide_ext = '.tif'
		ood_name_list=['TCGA_lung_ood','TCGA_breast_ood']
		ood_csv_path_list ={'TCGA_lung_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_lung_ood.csv',
					  'TCGA_breast_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_breast_ood.csv'}
	elif args.experiment_mask in ['RCC','RCC_conch']:
		ood_name_list=['RCC_ood', 'RCC_Normal']
		ood_csv_path_list =['./dataset_csv/tumor_subtyping_dummy_clean_RCC_ood.csv',
					  './dataset_csv/tumor_subtyping_dummy_clean_RCC_Normal.csv']


	m_list=[50,20,10,5,0.1,0.01]#最多5个参数
	args.split_dir = os.path.join('./splits', args.task+'_{}_{}'.format(args.experiment_mask,(args.label_frac*100))) 
	ood_dataset_splits(args.split_dir,args,ood_name_list,ood_csv_path_list)



	datasets=[]#10个训练集与两个OOD数据集
	for i in range(args.k):
		ood_csv_path='{}/ood_detection_splits_{}.csv'.format(args.split_dir, i)
		id_dataset = Dataset_All_Bags_selection(args.id_csv_path,args.id_slide_id_csv,ood_csv_path,mode = 'val')
		datasets.append((id_dataset,i))
	
	for ood_name in ood_name_list:
		if ood_name == 'RCC_Normal':
			#来自于两部分
			ood_csv_path='{}/ood_detection_splits_{}.csv'.format(args.split_dir,'RCC_normal_ffpe')
			ood_dataset = Dataset_All_Bags_selection(args.id_csv_path,args.id_slide_id_csv,ood_csv_path,mode = 'val')
			datasets.append((ood_dataset,ood_name))
			ood_csv_path='{}/ood_detection_splits_{}.csv'.format(args.split_dir,'RCC_normal_frozen')
			args.ood_csv_path = os.path.join(args.csv_path,'RCC_ood','process_list_autogen.csv')
			args.ood_slide_id_csv = os.path.join(args.slide_id_csv,'%s_dummy_clean_%s.csv'%(args.task,'RCC_ood'))
			ood_dataset = Dataset_All_Bags_selection(args.ood_csv_path,args.ood_slide_id_csv,ood_csv_path,mode = 'val')
			datasets.append((ood_dataset,ood_name))
		else:
			ood_csv_path='{}/ood_detection_splits_{}.csv'.format(args.split_dir,ood_name)
			args.ood_csv_path = os.path.join(args.csv_path,ood_name,'process_list_autogen.csv')
			args.ood_slide_id_csv = os.path.join(args.slide_id_csv,'%s_dummy_clean_%s.csv'%(args.task,ood_name))
			ood_dataset = Dataset_All_Bags_selection(args.ood_csv_path,args.ood_slide_id_csv,ood_csv_path,mode = 'val')
			datasets.append((ood_dataset,ood_name))



	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))


	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	is_parallel = False
	if is_parallel:#False:#True:
		args.batch_size=2048
		model = nn.DataParallel(model)
	else:
		torch.cuda.set_device(1)
	_ = model.eval()
	model = model.to(device)
	#total=id_wsi+ood_wsi
	#total = len(bags_dataset)
	total=0
	len_dataset=[]
	for item, (dataset,ood_name) in enumerate(datasets):
		print('第%d个验证集有%d个wsi'%(item,len(dataset)))
		len_dataset.append(len(dataset))
		total+=len(dataset)
	
	#从这里开始修改：注意：如果ID数据集出现过这个wsi的特征提取文件，则跳过，仅仅针对ID数据
	loader_kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}
	for item, (dataset,ood_name) in enumerate(datasets):
		for bag_candidate_idx in tqdm(range(len(dataset))):
			file_name_with_ext = os.path.basename(dataset[bag_candidate_idx])
			slide_id = file_name_with_ext.split(args.slide_ext)[0]
		    #slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			bag_name = slide_id+'.h5'
			if item <args.k:
				h5_file_path = os.path.join(args.id_data_h5_dir, 'patches', bag_name)
			else:
				if ood_name == 'RCC_Normal':
					args.ood_data_h5_dir = os.path.join(args.data_h5_dir, 'RCC') 
					h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', bag_name)
					if not os.path.exists(h5_file_path):
						args.ood_data_h5_dir = os.path.join(args.data_h5_dir, 'RCC_ood') 
						h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', bag_name)
				else:
					args.ood_data_h5_dir = os.path.join(args.data_h5_dir, ood_name)
					h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', bag_name)

			slide_file_path = dataset[bag_candidate_idx]	

			#slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
			if item == 0:
				print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
			else:
				print('\nprogress: {}/{}'.format(sum(len_dataset[:item])+bag_candidate_idx, total))
			print(slide_id)
			if item<args.k:
				output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			else:
				args.ood_feat_dir=os.path.join(args.ood_feat_dir_temp,'%s_perturbation'%(ood_name))
				os.makedirs(args.ood_feat_dir, exist_ok=True)
				os.makedirs(os.path.join(args.ood_feat_dir, 'pt_files'), exist_ok=True)
				os.makedirs(os.path.join(args.ood_feat_dir, 'h5_files'), exist_ok=True)
				ood_dest_files = os.listdir(os.path.join(args.ood_feat_dir, 'pt_files'))
				dest_files=ood_dest_files
				output_path = os.path.join(args.ood_feat_dir, 'h5_files', bag_name)
			
			if not args.no_auto_skip and slide_id+'.pt' in dest_files:
				print('skipped {}'.format(slide_id))
				continue
			#dest_files = os.listdir(os.path.join(args.feat_dir, 'h5_files'))
			#if not args.no_auto_skip and slide_id+'.h5' in dest_files:
			#	print('skipped {}'.format(slide_id))
			#	continue 
			
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			dataset_patch = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

			loader = DataLoader(dataset=dataset_patch, batch_size=args.batch_size, **loader_kwargs)
			output_file_path = compute_w_loader(output_path, loader = loader, model = model,args = args, verbose = -1,m_list=m_list)

			time_elapsed = time.time() - time_start
			t0=time.time()
			num_par=len(m_list)*4
			with h5py.File(output_file_path, "r") as file:
				for item_mag, magnitude in enumerate(m_list):
					for rotation in range(4):
						features = file['features_%d_%d'%(item_mag,rotation)][:]#维度：n*1024 n表示一个wsi中patch的数量
						if item_mag == 0 and rotation == 0:	
							print('features size: ', features.shape)
							print('coordinates size: ', file['coords'].shape)
							feature_chunnel,feature_shape=features.shape
							break
			features_pertude = torch.empty(num_par, feature_chunnel, feature_shape)#此时维度为(len(m_list)*num_par,n,1024)
			with h5py.File(output_file_path, "r") as file:
				for item_mag, magnitude in enumerate(m_list):
					for rotation in range(4):
						features = file['features_%d_%d'%(item_mag,rotation)][:]#维度：n*1024 n表示一个wsi中patch的数量
						features = torch.from_numpy(features)
						features_pertude[item_mag]=features
			t1=time.time()
			print('\ncomputing features for {} took {} s, and concatenating features took {} s'.format(output_file_path, time_elapsed,t1-t0))
			print('pertude features size: ', features_pertude.shape)
			bag_base, _ = os.path.splitext(bag_name)
			if item <args.k:
				torch.save(features_pertude, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'), pickle_protocol=4)
			else:
				torch.save(features_pertude, os.path.join(args.ood_feat_dir, 'pt_files', bag_base+'.pt'), pickle_protocol=4)






