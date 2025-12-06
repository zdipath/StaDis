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

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, args, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			
			if args.model_name == 'conch_v1':
				features = model.encode_image(batch, proj_contrast=False, normalize=False)
			else:
				features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
#需要修改的
parser.add_argument('--experiment_mask', type = str,default='TCGA_lung_ood',
					choices=['RCC','RCC_ood','C16','TCGA_lung_ood','TCGA_breast_ood','RCC_conch','RCC_ood_conch','C16_conch','TCGA_lung_ood_conch','TCGA_breast_ood_conch'],
					help='name about experiment')
parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50_trunc', 'uni_v1', 'conch_v1','resnet50'])

#default
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--data_h5_dir', type=str, default='./data/clam_format_patch/')
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default='./data/RCC_ood/process_list_autogen.csv')
parser.add_argument('--feat_dir', type=str, default='./data/')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	torch.cuda.set_device(1)
	#CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py  
	args.csv_path=os.path.join(args.data_h5_dir,args.experiment_mask,'process_list_autogen.csv')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	args.data_h5_dir = os.path.join(args.data_h5_dir,args.experiment_mask)
	args.feat_dir = os.path.join(args.feat_dir,'clam_format_feature_%s'%(args.model_name),args.experiment_mask)

	if args.experiment_mask in  ['C16','C16_conch']:
		args.slide_ext = '.tif'




	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		file_name_with_ext = os.path.basename(bags_dataset[bag_candidate_idx])
		slide_id = file_name_with_ext.split(args.slide_ext)[0]
		#slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

		slide_file_path = bags_dataset[bag_candidate_idx]
		#slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model,args = args, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



