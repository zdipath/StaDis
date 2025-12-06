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
import sklearn.covariance
import faiss
def covaraiance_mean(model,num_classes, train_loader,dir_name,args):
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)#计算原始数据的协方差矩阵而不是中心化后的数据
    num_sample_per_class = np.empty(num_classes)#每个类别的样本数量
    num_sample_per_class.fill(0)
    t0=time.time()
    num_breakpoint=0
    
    concat_parts = [[] for _ in range(num_classes)]
    list_features = [[] for _ in range(num_classes)]
    for item_data,batch_data in enumerate(train_loader):
        #if item_data==300:
        #    break
        data, target, coords, id = batch_data#[21816, 1024]
        data = data.cuda()
        #print(data.shape[0])
        if data.shape[0]>10000 and args.model_type == 'transmil':
            indices = torch.randperm(data.size(0))[:10000]
            data = data[indices]
        with torch.no_grad():  # 禁用梯度计算，减少显存占用
            _, _, _, wsi_feature, _ = model(data)
            label = target[0]
            concat_parts[label].append(wsi_feature.view(1, -1))  

    for i in range(num_classes):
        list_features[i]=torch.cat(concat_parts[i],dim=0)
    temp_file_name_covaraiance = '%s/covaraiance.txt'%(dir_name)
    temp_file_name_mean = '%s/mean.txt'%(dir_name)
    g = open(temp_file_name_covaraiance, 'w')
    f = open(temp_file_name_mean, 'w')
    sample_class_mean = []#存储每个类别的特征均值

    temp_list = torch.zeros(num_classes,list_features[0].shape[1]).cuda() #类别
    for j in range(num_classes):
        feature_mean=list_features[j]
        temp_list[j] = feature_mean.mean(dim=0, keepdim=True)
        sample_class_mean.append(temp_list[j])
        f.write(' '.join(map(str, temp_list[j].tolist())))
        f.write('\n')
  
    for i in range(num_classes):
        #list_features[k][i] = list_features[k][i].to('cuda')
        sample_class_mean_cpu=sample_class_mean[i]
        if i == 0:
            X = list_features[i] - sample_class_mean_cpu#都放到cpu上运算
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean_cpu), 0)
        # find inverse            
    group_lasso.fit(X.cpu().numpy())#X.cpu().numpy()
    temp_precision = group_lasso.precision_
    temp_precision = torch.from_numpy(temp_precision).float().cuda()



    temp_precision_numpy = temp_precision.cpu().numpy()
    temp_precision_cuda = torch.from_numpy(temp_precision_numpy).float().cuda()
    temp_precision_list = temp_precision_cuda.tolist()
    #temp_precision_list=[[1,2,3],[4,5,6],[7,8,9]]
    temp_precision_str = ' '.join(map(str, temp_precision_list))
        
    g.write(temp_precision_str+'\n')
    precision=temp_precision
    f.close()
    g.close()
    return# sample_class_mean, precision
def Mahalanobis_score(data, coords, model,id,M_list,temperature,sample_mean,covariation,n_classes,args,out_flag):
    model.eval()
    Mahalanobis = []
    t0=time.time()
    
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    data.requires_grad = True
    _, _, _, wsi_feature, _ = model(data)
    sample_mean = torch.tensor(sample_mean).cuda()
    out_features=wsi_feature
    gaussian_score = 0
    for i in range(n_classes):
        batch_sample_mean = sample_mean[i]#第 层特征、第 个类的均值
        #batch_sample_mean = torch.tensor(batch_sample_mean).cuda()
        zero_f = out_features.data - batch_sample_mean#减去均值
        term_gau = -0.5*torch.mm(torch.mm(zero_f, covariation), zero_f.t()).diag()#mm是相乘，precision是协方差，表示了样本之间的马氏距离
        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)#按照列进行拼接
        
    sample_pred = gaussian_score.max(1)[1]#对应着一个batch的结果
    batch_sample_mean = sample_mean.index_select(0, sample_pred)#index_select 方法允许在指定维度上根据索引值选择元素   目的是从样本均值中选择特定样本的均值
    zero_f = out_features -  (batch_sample_mean)
    pure_gau = -0.5*torch.mm(torch.mm(zero_f, covariation), zero_f.t()).diag()
    model.zero_grad()
    loss = torch.mean(-pure_gau)
    loss.backward()
        
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    #直接在feature层面的扰动
    features=[]
    for magnitude in M_list:
        feature=torch.add(data.data, gradient,alpha= -magnitude)
        features.append(feature)
    #features=make_FGSM_noise_feature(data,id,args,gradient,M_list,out_flag)
    for feature in features:
        if feature.shape[0]>10000 and args.model_type == 'transmil':
            indices = torch.randperm(feature.size(0))[:10000]
            feature = feature[indices]
        _, _, _, noise_wsi_feature, _ = model(feature)
        
        noise_gaussian_score = 0
        for i in range(n_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = noise_wsi_feature - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, covariation), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      
        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.detach().cpu().numpy())
        
    
    #print('花费了%.2f秒，'%(time.time()-t0))
    #print(num_breakpoint,time.time()-t0)
    t0=time.time()
    return Mahalanobis
def make_FGSM_noise_feature(data,id,args,gradient,M_list,out_flag):
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    _ = model.eval()
    loader_kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}

    if out_flag:#ID
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', id+'.h5')#从之前的pt中获取corrds
        csv_path = args.csv_path
    else:
        h5_file_path = os.path.join(args.ood_data_h5_dir, 'patches', id+'.h5')
        csv_path = args.ood_csv_path
    df = pd.read_csv(csv_path)
    slide_file_path = df[df['slide_id'].str.contains(id)]['slide_id'].iloc[0]
    #35服务器操作
    time_start = time.time()
    wsi = openslide.open_slide(slide_file_path)
    dataset_patch = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)
    loader = DataLoader(dataset=dataset_patch, batch_size=args.batch_size, **loader_kwargs)
    features = compute_w_loader(M_list,gradient, loader = loader, model = model, verbose = 1)
    time_elapsed = time.time() - time_start
    print('\ncomputing features for {} took {} s'.format(id, time_elapsed))
    return features

def compute_w_loader(M_list, gradient, loader, model, verbose = 0):
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))
	mode = 'w'
	features=[]
	for count, data in enumerate((loader)):
		with torch.inference_mode():	
			batch = data['img'].cuda()
			coords = data['coord'].numpy().astype(np.int32)
			for magnitude in M_list:
				temp_x_noise=torch.add(batch.data, gradient,alpha= -magnitude)
				feature = model(temp_x_noise).cpu().numpy().astype(np.float32)
				features.append(feature)



	
	return features


def MDS_val_score(sample_mean,covariation,net,test_loader,output_path,out_flag,M_list,T_list,n_classes,args,oodname=''):
    if out_flag == True:
        temp_file_name_val = '%s/Probability_MDS_val_In.txt'%(output_path)
    else:
        temp_file_name_val = '%s/Probability_MDS_val_Out_%s.txt'%(output_path,oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    g = open(temp_file_name_val, 'w')
    t0=time.time()
    for item_val,temperature in enumerate(T_list):
        out = np.empty((len(test_loader), len(M_list)))
        for item_data,batch_data in enumerate(test_loader):
            #if item_data==1:
            #    continue
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]#+'.h5'
            #noise_fea_path=os.path.join(perturbation_feature_path,'h5_files',id)
            data, target = data.cuda(), target.cuda()
            #根据编号来导入ood数据。
            temp_diff_Outputs=Mahalanobis_score(data, coords, net,id,M_list,temperature,sample_mean,covariation,n_classes,args,out_flag)#是一个列表
            score=np.array(temp_diff_Outputs)
            out[item_data]=score.reshape(1,-1)
                # break
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()

def MDS_test_score(sample_mean,covariation,net,test_loader,output_path,out_flag,magnitude,temperature,n_classes,args,oodname=''):
    if out_flag == True:
        temp_file_name_test = '%s/Probability_MDS_test_In_%s.txt'%(output_path,oodname)
    else:
        temp_file_name_test = '%s/Probability_MDS_test_Out_%s.txt'%(output_path,oodname)
    f = open(temp_file_name_test, 'w')
    t0=time.time()
    M_list=[magnitude]
    out = np.empty((len(test_loader), len(M_list)))
    for item_data,batch_data in enumerate(test_loader):
        data, target, coords, id = batch_data#[21816, 1024]
        id = id[0]#+'.h5'
        data, target = data.cuda(), target.cuda()
        #根据编号来导入ood数据。
        temp_diff_Outputs=Mahalanobis_score(data, coords, net,id,M_list,temperature,sample_mean,covariation,n_classes,args,out_flag)#是一个列表
        score=np.array(temp_diff_Outputs)   
        out[item_data]=score.reshape(1,-1)
            #break
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time() 
    for i in range(out.shape[0]):
        f.write("{}\n".format(out[i,0]))
    f.close()

def ODIN_score(data, coords, model,id,M_list,temperature,n_classes,args,out_flag):
    score = []
    criterion = nn.CrossEntropyLoss()
    t0=time.time()
    
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    data.requires_grad = True
    logits, _, _, _, _ = model(data)
    outputs = logits/temperature
    labels = outputs.data.max(1)[1]
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2#变为正负1
    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    Inputs=[]
    for magnitude in M_list:
        tempInputs=torch.add(data.data, gradient,alpha= -magnitude)
        Inputs.append(tempInputs)
    #features=make_FGSM_noise_feature(data,id,args,gradient,M_list,out_flag)
    for input in Inputs:
        if input.shape[0]>10000 and args.model_type == 'transmil':
            indices = torch.randperm(input.size(0))[:10000]
            input = input[indices]
        outputs, _, _, _, _ = model(input)
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out, dim=1)
        score.extend(soft_out.detach().cpu().numpy())
    return score



def ODIN_val_test_score(net,test_loader,output_path,out_flag,M_list,T_list,n_classes,args,oodname='', mode = 'val'):
    if mode == 'val':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_ODIN_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_ODIN_val_Out_%s.txt'%(output_path,oodname)
    elif mode == 'odin_test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_ODIN_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_val = '%s/Probability_ODIN_test_Out_%s.txt'%(output_path,oodname)
    elif mode == 'msp_test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_MSP_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_val = '%s/Probability_MSP_test_Out_%s.txt'%(output_path,oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    g = open(temp_file_name_val, 'w')
    t0=time.time()
    
    for item_val,temperature in enumerate(T_list):
        out = np.empty((len(test_loader), len(M_list)))
        for item_data,batch_data in enumerate(test_loader):
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]
            data, target = data.cuda(), target.cuda()
            temp_diff_Outputs=ODIN_score(data, coords, net,id,M_list,temperature,n_classes,args,out_flag)#是一个列表
            score=np.array(temp_diff_Outputs)
            out[item_data]=score.reshape(1,-1)   
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()

def React_score(data, coords, model,id,threshold_list,temperature,n_classes,args,out_flag):
    model.eval()
    score = []
    t0=time.time()
    data.requires_grad = True
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    method = 'energy'
    with torch.no_grad():
        for i, threshold in enumerate(threshold_list):
            logits = model.forward_threshold(data, threshold)
            if method == "energy":
                scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
            elif method=='msp':
                scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
            if i ==0:
                output=scores.reshape(-1,1)
            else:
                output=np.concatenate((output,scores.reshape(-1,1)),axis=1)
    return output

def EBO_score(data, coords, model,id,threshold_list,temperature,n_classes,args,out_flag):
    model.eval()
    
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    data.requires_grad = True
    logits, _, _, _, _ = model(data)
    with torch.no_grad():
        for i, threshold in enumerate(threshold_list):
            scores = temperature*torch.logsumexp((logits/temperature).cpu(), dim=1).numpy()
            if i ==0:
                output=scores.reshape(-1,1)
            else:
                output=np.concatenate((output,scores.reshape(-1,1)),axis=1)
    return output

def EBO_val_test_score(net,test_loader,output_path,out_flag,threshold_list,T_list,n_classes,args,oodname='', mode = 'val'):
    if mode == 'val':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_EBO_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_EBO_val_Out_%s.txt'%(output_path,oodname)
    elif mode == 'test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_EBO_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_val = '%s/Probability_EBO_test_Out_%s.txt'%(output_path,oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    t0=time.time()
    outs = np.empty((len(test_loader), len(T_list)*len(threshold_list)))
    
    for item_data,batch_data in enumerate(test_loader):
        for item_val,temperature in enumerate(T_list):
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]
            data, target = data.cuda(), target.cuda()
            score=EBO_score(data, coords, net,id,threshold_list,temperature,n_classes,args,out_flag)#是一个列表
            outs[item_data,item_val*len(threshold_list):item_val*len(threshold_list)+len(threshold_list)]=score.reshape(1,-1)
        
    print('总共花费时间%.2f秒'%(time.time()-t0))  
    t0=time.time()
    g = open(temp_file_name_val, 'w')
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()

def React_threshold(thresh_list,train_loader,model,network_name,output_path):
    threshold = []
    t1 = time.time()
    thres_path = os.path.join(output_path,'_'.join(map(str, thresh_list))+'.npy')
    if os.path.exists(thres_path):
        loaded_threshold_array = np.load(thres_path)
        return loaded_threshold_array.tolist()
    for thre in thresh_list:
        activation_log = []
        for item_data,batch_data in enumerate(train_loader):
            #if item_data==10:
            #    break
            data, target, coords, id = batch_data#[21816, 1024]
            data = data.cuda()
            if data.shape[0]>10000 and network_name == 'transmil':
                indices = torch.randperm(data.size(0))[:10000]
                data = data[indices]
            with torch.no_grad():  # 禁用梯度计算，减少显存占用
                _, _, _, wsi_feature, _ = model(data)
                activation_log.append(wsi_feature.data.cpu().numpy())
        activation_log = np.concatenate(activation_log, axis=0)
        threshold.append(np.percentile(activation_log.flatten(),thre*100))
    threshold_array = np.array(threshold)
    np.save(thres_path, threshold_array)
    print('Computing threshold: %.2f'%(time.time()-t1))
    return threshold



def uncertainty_score(net,test_loader,output_path,out_flag,T_list,n_classes,args,oodname='', mode = 'val', method = 'M-heads',OOD_score_way = 'mean'):
    #OOD_score_way = 'mean'#'variance' # 'mean'
    if mode == 'val':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_%s_val_In.txt'%(output_path, method)
        else:
            temp_file_name_val = '%s/Probability_%s_val_Out_%s.txt'%(output_path, method, oodname)
    elif mode == 'test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_%s_test_In_%s.txt'%(output_path, method, oodname)
        else:
            temp_file_name_val = '%s/Probability_%s_test_Out_%s.txt'%(output_path, method, oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    g = open(temp_file_name_val, 'w')
    if method == 'MCDropout':
        net.train()
    else:
        net.eval()
    t0=time.time()
    for item_val,temperature in enumerate(T_list):
        out = np.empty((len(test_loader), 1))
        for item_data,batch_data in enumerate(test_loader):
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]
            data, target = data.cuda(), target.cuda()
            if data.shape[0]>10000 and args.model_type in ['transmil_ensemble','transmil_dropout']:
                indices = torch.randperm(data.size(0))[:10000]
                data = data[indices]
            with torch.no_grad():
                if method == 'M-heads':
                    logits1, logits2, logits3, logits4, logits5, Y_prob, Y_hat, _, _ = net(data)
                    outputs_list = [logits1, logits2, logits3, logits4, logits5]
                else:
                    outputs_list = []
                    for i in range(5):
                        logits, Y_prob, Y_hat, _, _ = net(data)
                        outputs_list.append(logits)
                logits_list = []
                for outputs in outputs_list:
                    outputs = outputs / temperature
                    soft_out = F.softmax(outputs, dim=1)
                    #soft_out, _ = torch.max(soft_out, dim=1)
                    logits_list.append(soft_out)
                if OOD_score_way == 'variance':
                    stacked_outputs = torch.stack(logits_list)
                    sample_variances = stacked_outputs.var(dim=0, unbiased=False)
                    mean_variance = sample_variances.mean(dim=1)      
                elif OOD_score_way == 'mean':
                    mean = torch.stack(logits_list, dim=0).mean(dim=0, keepdim=True).squeeze(0)
                    epsilon = 1e-8  # 防止 log(0) 导致的问题
                    mean_variance = - torch.sum(mean * torch.log(mean + epsilon), dim=1)
                out[item_data,0] = - mean_variance.detach().cpu().numpy()
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()


def React_val_test_score(net,test_loader,output_path,out_flag,threshold_list,T_list,n_classes,args,oodname='', mode = 'val'):
    if mode == 'val':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_React_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_React_val_Out_%s.txt'%(output_path,oodname)
    elif mode == 'test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_React_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_val = '%s/Probability_React_test_Out_%s.txt'%(output_path,oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    g = open(temp_file_name_val, 'w')
    t0=time.time()
    for item_val,temperature in enumerate(T_list):
        out = np.empty((len(test_loader), len(threshold_list)))
        for item_data,batch_data in enumerate(test_loader):
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]
            data, target = data.cuda(), target.cuda()
            score=React_score(data, coords, net,id,threshold_list,temperature,n_classes,args,out_flag)#是一个列表
            out[item_data]=score.reshape(1,-1)
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()




normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def normal_fea(model,num_classes, train_loader,dir_name,args):
    model.eval()
    t0 = time.time()
    for item_data,batch_data in enumerate(train_loader):
        data, target, coords, id = batch_data#[21816, 1024]
        data = data.cuda()
        if data.shape[0]>10000 and args.model_type == 'transmil':
            indices = torch.randperm(data.size(0))[:10000]
            data = data[indices]
        with torch.no_grad():  # 禁用梯度计算，减少显存占用
            _, _, _, wsi_feature, _ = model(data)
            feature_dim = wsi_feature.shape[1]
            break
    total_batches = len(train_loader)
    activation_log = np.zeros((total_batches, feature_dim), dtype=np.float32)
    for item_data,batch_data in enumerate(train_loader):
        #if item_data==8:
        #    break
        data, target, coords, id = batch_data#[21816, 1024]
        data = data.cuda()
        if data.shape[0]>10000 and args.model_type == 'transmil':
            indices = torch.randperm(data.size(0))[:10000]
            data = data[indices]
        with torch.no_grad():  # 禁用梯度计算，减少显存占用
            _, _, _, wsi_feature, _ = model(data) #1*1024
            activation_log[item_data] = normalizer(wsi_feature.data.cpu().numpy())
    #activation_log = np.concatenate(activation_log, axis=0)#.reshape(len(activation_log),activation_log[0].shape[1]) 
    index = faiss.IndexFlatL2(wsi_feature.shape[1])
    index.add(activation_log)
    print('the time of normal training fea is：%s'%(time.time()-t0))
    return index



def KNN_val_test_score(dis_index,net,test_loader,output_path,out_flag,K_list,T_list,n_classes,args,oodname='',mode = 'val'):
                                        
    if mode == 'val':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_KNN_val_In.txt'%(output_path)
        else:
            temp_file_name_val = '%s/Probability_KNN_val_Out_%s.txt'%(output_path,oodname)
    elif mode == 'test':
        if out_flag == True:
            temp_file_name_val = '%s/Probability_KNN_test_In_%s.txt'%(output_path,oodname)
        else:
            temp_file_name_val = '%s/Probability_KNN_test_Out_%s.txt'%(output_path,oodname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    g = open(temp_file_name_val, 'w')
    t0=time.time()
    for item_val,temperature in enumerate(T_list):
        out = np.empty((len(test_loader), len(K_list)))
        for item_data,batch_data in enumerate(test_loader):
            data, target, coords, id = batch_data#[21816, 1024]
            id = id[0]
            data, target = data.cuda(), target.cuda()
            score=KNN_score(data,dis_index, net,id,K_list,temperature,n_classes,args,out_flag)#是一个列表
            out[item_data]=score.reshape(1,-1)
        if item_val==0:
            outs=out
        else:
            outs=np.concatenate((outs,out),axis=1)
    print('总共花费时间%.2f秒'%(time.time()-t0))    
    t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()

def KNN_score(data, dis_index, model,id,threshold_list,temperature,n_classes,args,out_flag):
    model.eval()
    if data.shape[0]>10000 and args.model_type == 'transmil':
        indices = torch.randperm(data.size(0))[:10000]
        data = data[indices]
    _, _, _, wsi_feature, _ = model(data)
    feature_normed = normalizer(wsi_feature.data.cpu().numpy())
    for i, threshold in enumerate(threshold_list):
        threshold = int(threshold)
        D, _ = dis_index.search(
                    feature_normed,
                    threshold,
                    )
        scores = -D[:, -1]
        if i ==0:
            output=scores.reshape(-1,1)
        else:
            output=np.concatenate((output,scores.reshape(-1,1)),axis=1)
    return output
