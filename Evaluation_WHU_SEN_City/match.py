import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scipy.io import savemat
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage.feature import match_descriptors
import scipy.io as scio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--feature_name", type=str, default='WHU_SEN_City',help='Name of feature')
parser.add_argument("--subsets", type=str, default='VIS_SAR',help="VIS_SAR")
parser.add_argument("--nums_kp", type=int, default=-1, help="Number of feature for evluation")
parser.add_argument("--vis_flag", type=bool, default=True,help="Visualization flag")
args = parser.parse_args() 

import argparse
bf = cv2.BFMatcher(crossCheck=True)

lm_counter = 0
MIN_MATCH_COUNT = 5
num_black_list = 0

subsets = [args.subsets]

if args.nums_kp < 0:
    nums_kp = [1024,2048,4096]
else:
    nums_kp = [args.nums_kp]
feature_name = args.feature_name
vis_flag = args.vis_flag


for subset in subsets:
    subset_path = os.path.join(SCRIPT_DIR,subset)
    dirlist = os.listdir(subset_path)
    if 'test' in dirlist:
        imgs = os.listdir(os.path.join(subset_path,'test','VIS'))
    else:
        continue
    print(subset)
    filepath1 = os.path.join(subset_path,'test',subset.split('_')[0])
    filepath2 = os.path.join(subset_path,'test',subset.split('_')[1])
    for num in [1024,2048,4096]:
    # for num in [2048]:
        N_k1 = []
        N_k2 = []
        N_corr = []
        N_corretmatches = []
        N_in_corretmatches = []
        N_k1_ol = []
        N_k2_ol = []
        N_corr_thres = []
        N_corretmatches_thres = []
        N_matches = []  
        RMSE_Ncm_list = []
        image_list = sorted(os.listdir(filepath1))
        img_list_whitelist = []
        progress_bar = tqdm(range(len(image_list)))
        for id in progress_bar:
            # i=id+1
            img_list_whitelist.append(image_list[id])
            imgpath1 = os.path.join(filepath1, image_list[id])
            imgpath2 = os.path.join(filepath2, image_list[id])
            image1 = np.array(Image.open(imgpath1).convert('RGB'))
            image2 = np.array(Image.open(imgpath2).convert('RGB'))
            ff = image_list[id].replace('.png','.features.mat')
            feats = scio.loadmat(os.path.join(SCRIPT_DIR,'features',subset,feature_name,ff))
            desc1 = feats['desc1']
            desc2 = feats['desc2']
            kp1 = feats['kp1'][:,0:2]
            kp2 = feats['kp2'][:,0:2]
            try:
                suffix = '.12'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']
                ones = np.ones_like(image1)
                mask = cv2.warpPerspective(ones,H,[ones.shape[1],ones.shape[0]])
                mask_1 = mask>0.5
                mask_1 = mask_1*1.0
                mask = cv2.warpPerspective(mask_1,np.linalg.inv(H),[mask.shape[1],mask.shape[0]])
                mask_2 = mask>0.5
                ones = np.ones([np.size(kp2,0),1])
                kp_2_warped = np.hstack([kp2,ones])
                kp_2_warped = H @ kp_2_warped.transpose()
                kp_2_warped = kp_2_warped/kp_2_warped[2,:]
                kp_2_warped = kp_2_warped[0:2,:].transpose()
                kp_1_warped = kp1
            except:
                suffix = '.21'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']
                ones = np.ones_like(image1)
                mask = cv2.warpPerspective(ones,H,[ones.shape[1],ones.shape[0]])
                mask_2 = mask>0.5
                mask_2 = mask_2*1.0
                mask = cv2.warpPerspective(mask_2,np.linalg.inv(H),[mask.shape[1],mask.shape[0]])
                mask_1 = mask>0.5
                ones = np.ones([np.size(kp1,0),1])
                kp_1_warped = np.hstack([kp1,ones])
                kp_1_warped = H @ kp_1_warped.transpose()
                kp_1_warped = kp_1_warped/kp_1_warped[2,:]
                kp_1_warped = kp_1_warped[0:2,:].transpose()
                kp_2_warped = kp2
            N_k1.append(kp1[0:num].shape[0])
            N_k2.append(kp2[0:num].shape[0])

            overlap1 = 0
            for kp in kp1[0:num]:
                x = int(kp[0]+0.5)
                y = int(kp[1]+0.5)
                if mask_1[(y,x)].sum(axis=-1) > 0.5:
                    overlap1 += 1
            N_k1_ol.append(overlap1)

            overlap2 = 0
            for kp in kp2[0:num]:
                x = int(kp[0]+0.5)
                y = int(kp[1]+0.5)
                if mask_2[((y,x))].sum(axis=-1) > 0.5:
                    overlap2 += 1
            N_k2_ol.append(overlap2)
            
            kp_1_warped_  = kp_1_warped[0:num][:,:2].reshape(-1,1,2)
            kp_2_warped_ = kp_2_warped[0:num][:,:2].reshape(1,-1,2)
            dist_k = ((kp_1_warped_ - kp_2_warped_)**2).sum(axis=2)
            
            matches = match_descriptors(desc1[0:num], desc2[0:num], cross_check=False)
            num_crosscheck_matches = matches.shape[0]  
            N_matches.append(num_crosscheck_matches)  
            keypoints_left = kp_1_warped[0:num][matches[:, 0], : 2]
            keypoints_left_raw = kp1[0:num][matches[:, 0], : 2]
            keypoints_right = kp_2_warped[0:num][matches[:, 1], : 2]
            keypoints_right_raw = kp2[0:num][matches[:, 1], : 2]
            
            dif = (keypoints_left - keypoints_right)
            dist_m = dif[:, 0]**2 + dif[:, 1]**2
            for thres in range(1,11):
                
                n_corr = ((dist_k<=thres**2).sum(axis=1)>0.9).sum()
                N_corr_thres.append(n_corr.item())
                if thres==3:
                    N_corr.append(n_corr.item())
                
                inds = dist_m<=thres**2
                N_corretmatches_thres.append(inds.sum())
                if thres==3:
                    N_corretmatches.append(inds.sum())

                    if dist_m.size > 0 and np.any(inds):
                        rmse_ncm = float(np.sqrt(np.mean(dist_m[inds])))
                    else:
                        rmse_ncm = np.nan
                    RMSE_Ncm_list.append(rmse_ncm)
                
                    if vis_flag and thres == 3:
                        
                        n_corr_mask_vis = (dist_k <= thres**2).sum(axis=1) > 0.9
                        n_corr_mask_sar = (dist_k <= thres**2).sum(axis=0) > 0.9
                        pts_corr_vis = kp1[0:num][n_corr_mask_vis]
                        pts_corr_sar = kp2[0:num][n_corr_mask_sar]

                        offset = image1.shape[1]
                        img_cat = np.concatenate((image1, image2), axis=1)

                        for pt in pts_corr_vis:
                            cv2.circle(img_cat, tuple(np.round(pt).astype(int)), 3, (255, 0, 0), -1)
                        for pt in pts_corr_sar:
                            pt_offset = tuple(np.round(pt + np.array([offset, 0])).astype(int))
                            cv2.circle(img_cat, pt_offset, 2, (255, 0, 0), -1)

                        for idx in np.where(dist_m <= thres**2)[0]:
                            pt1 = tuple(np.round(keypoints_left_raw[idx]).astype(int))
                            pt2 = tuple(np.round(keypoints_right_raw[idx]).astype(int) + np.array([offset, 0]))
                            cv2.circle(img_cat, pt1, 2, (0, 255, 0), -1)
                            cv2.circle(img_cat, pt2, 2, (0, 255, 0), -1)
                            cv2.line(img_cat, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

                        vis_save_dir = os.path.join(SCRIPT_DIR, 'results', subset, feature_name, 'vis')
                        os.makedirs(vis_save_dir, exist_ok=True)
                        image_basename = os.path.splitext(image_list[id])[0]
                        vis_path = os.path.join(vis_save_dir, f'match_vis_{image_basename}_{num}.png')
                        cv2.imwrite(vis_path, img_cat)

        N_corr_thres = np.array(N_corr_thres)
        N_corr = np.array(N_corr)*1.0
        N_k1 = np.array(N_k1)*1.0
        N_k2 = np.array(N_k2)*1.0
        N_k1_ol = np.array(N_k1_ol)
        N_k2_ol = np.array(N_k2_ol)
        N_corretmatches = np.array(N_corretmatches)*1.0
        N_corretmatches_thres = np.array(N_corretmatches_thres)
        RR = N_corr*1.0/np.array([N_k1,N_k2]).min(axis=0)
        mask_zero = N_k1_ol<0.1
        N_k1_ol_temp = N_k1_ol
        N_k1_ol_temp[mask_zero]=1
        mask_zero = N_k2_ol<0.1
        N_k2_ol_temp = N_k2_ol
        N_k2_ol_temp[mask_zero]=1
        MS = (N_corretmatches*1.0/N_k1_ol+N_corretmatches*1.0/N_k2_ol)/2
        rmse_ncm_mean = float(np.nanmean(np.array(RMSE_Ncm_list))) if len(RMSE_Ncm_list) else float('nan')
        cmr = np.mean(N_corretmatches) / N_corr.mean() if N_corr.mean() > 0 else 0
        if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name)):
            os.makedirs(os.path.join(SCRIPT_DIR,'results',subset,feature_name))
        savemat(os.path.join(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match_result_{}.mat'.format(num))),{'N_corr':N_corr,'N_k1':N_k1,'N_k2':N_k2,'N_correctmatches':N_corretmatches,
                                                                                'N_k1_ol':N_k1_ol,'N_k2_ol':N_k2_ol,'N_correctmatches_thres':N_corretmatches_thres,
                                                                                'N_corr_thres':N_corr_thres})
        print('Number of infrared keypoints: %f.' % np.mean(N_k1))
        print('Number of visible keypoints: %f.' % np.mean(N_k2))
        print('Number of correspondence: %f.' % N_corr.mean())
        print('Number of correct matches: %f.' % np.mean(N_corretmatches))
        print('CMR: {:.4f}.'.format(cmr))
        print('RMSE: {:.3f} px'.format(rmse_ncm_mean))
        print('RR: {}.'.format(RR.mean()))
        print('MS: {}.'.format(MS.mean()))

        log_file_path = os.path.join(SCRIPT_DIR, 'results', subset, feature_name, 'match_log.txt')
        if os.path.exists(log_file_path):
            print('[Warning] Log file already exists.')
        log_file = open(log_file_path, 'a+')
        log_file.write('Number of infrared keypoints: %f.\n' % np.mean(N_k1))
        log_file.write('Number of visible keypoints: %f.\n' % np.mean(N_k2))
        log_file.write('Number of correspondence: %f.\n' % N_corr.mean())
        log_file.write('Number of correct matches: %f.\n' % np.mean(N_corretmatches))
        log_file.write('CMR: {:.4f}.\n'.format(cmr))
        log_file.write('RMSE: {:.3f} px\n'.format(rmse_ncm_mean))
        log_file.write('RR: {}.\n'.format(RR.mean()))
        log_file.write('MS: {}.\n'.format(MS.mean()))
        log_file.close()

