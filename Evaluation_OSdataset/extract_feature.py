import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from eva_lib.model import MMNet

import sys

import scipy.io as scio
from copy import deepcopy
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

AP = nn.AvgPool2d(9, stride=1, padding=4).cuda()
MP = nn.AvgPool2d(9, stride=1, padding=4).cuda()

def load_network(model_fn):
    print("[INFO] Loading model from:", model_fn)
    checkpoint = torch.load(model_fn)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return model.eval()

class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rep_thr=0.6):
        super(NonMaxSuppression, self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr

    def forward(self, repeatability):
        maxima = (repeatability == self.max_filter(repeatability))
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima * 0
        border_mask[:, :, 10:-10, 10:-10] = 1
        maxima = maxima * border_mask
        print("[INFO] NMS points count:", maxima.sum().item())
        return maxima.nonzero().t()[2:4]

def extract_multiscale(net, img, detector, image_type,
                        scale_f=2 ** 0.25, min_scale=0.0,
                        max_scale=1, min_size=256,
                        max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    B, three, H, W = img.shape
    assert B == 1 and three == 3
    s = 1.0
    X, Y, S, C, Q, D = [], [], [], [], [], []
    scale_counter = 0
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose:
                print(f"[INFO] Extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            print(f"[INFO] Processing scale level {scale_counter+1}...")
            mask_extra = (MP((img > 1e-12).sum(dim=1, keepdim=True).float()) > 1e-5).float()
            for ii in range(3):
                mask_extra = (AP(mask_extra) > 0.99).float()
            img_t = (img - img.mean(dim=[-1, -2], keepdim=True)) / img.std(dim=[-1, -2], keepdim=True)
            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img_t)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img_t)
            mask = repeatability * 0
            mask[:, :, args.border:-args.border, args.border:-args.border] = 1
            repeatability = repeatability * mask * mask_extra
            y, x = detector(repeatability)
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            Q.append(q)
            D.append(d)
            print(f"[INFO] Detected {n} keypoints at scale x{s:.2f}")
            scale_counter += 1
        s /= scale_f
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
    torch.backends.cudnn.benchmark = old_bm
    Y = torch.cat(Y)
    X = torch.cat(X)
    scores = torch.cat(Q)
    XYS = torch.stack([X, Y], dim=-1)
    D = torch.cat(D)
    print(f"[INFO] Total keypoints collected: {len(XYS)}")
    return XYS, D, scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--subsets", type=str, default='VIS_SAR', help='VIS_SAR')
    parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
    parser.add_argument("--model", type=str, default='/checkpoints/OSdataset/OSdataset.pth', help='model path')
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--border", type=float, default=5)
    parser.add_argument("--repeatability-thr", type=float, default=0.4)
    parser.add_argument("--gpu", type=int, default=0, help='use -1 for CPU')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)

    args.subsets = [args.subsets]
    feature_name = 'OSdataset'  # 'Name of feature'
    net = load_network(args.model).cuda()
    detector = NonMaxSuppression(rep_thr=args.repeatability_thr)

    os.makedirs(os.path.join(SCRIPT_DIR, 'features', args.subsets[0], feature_name), exist_ok=True)
    type1 = args.subsets[0].split('_')[0]
    type2 = args.subsets[0].split('_')[1]

    for subset in args.subsets:
        file_path = os.path.join(SCRIPT_DIR, subset, 'test', type1)
        imgs = sorted([f for f in os.listdir(file_path) if f.endswith('.png')])
        print(f"[INFO] Extracting features for {len(imgs)} images in {subset}/{type1}")
        for i, img_name in enumerate(tqdm(imgs, desc=f"Processing {subset}")):
            print(f"[INFO] ({i+1}/{len(imgs)}) Processing image: {img_name}")
            img_path1 = os.path.join(SCRIPT_DIR, subset, 'test', type1, img_name)
            image = Image.open(img_path1).convert('RGB')

            image = TF.to_tensor(image).unsqueeze(0).cuda()
            xys, desc, scores = extract_multiscale(net, image, detector, '1',

                scale_f=args.scale_f, min_scale=args.min_scale, max_scale=args.max_scale,
                min_size=args.min_size, max_size=args.max_size, verbose=False)
            idxs = scores.topk(min(len(scores), args.num_features))[1]
            kp1, desc1 = xys[idxs].cpu().numpy(), desc[idxs].cpu().numpy()

            img_path2 = os.path.join(SCRIPT_DIR, subset, 'test', type2, img_name)
            image = Image.open(img_path2).convert('RGB')
            image = TF.to_tensor(image).unsqueeze(0).cuda()
            xys, desc, scores = extract_multiscale(net, image, detector, '2',
                scale_f=args.scale_f, min_scale=args.min_scale, max_scale=args.max_scale,
                min_size=args.min_size, max_size=args.max_size, verbose=False)
            idxs = scores.topk(min(len(scores), args.num_features))[1]
            kp2, desc2 = xys[idxs].cpu().numpy(), desc[idxs].cpu().numpy()

            out_path = os.path.join(SCRIPT_DIR, 'features', subset, feature_name, img_name.replace('.png', '.features.mat'))
            scio.savemat(out_path, {'desc1': desc1, 'kp1': kp1, 'desc2': desc2, 'kp2': kp2})
            print(f"[INFO] Saved features to {out_path}")
