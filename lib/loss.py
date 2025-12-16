
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia.geometry.transform as KGT
import kornia.filters as KF
import kornia.utils as KU
class MMLoss(nn.Module):
    def __init__(self, lam1=1,lam2=1,lam3=1, sample_n = 4096, input_size=192, sample_size=16, safe_radius_neg=7, safe_radius_pos=3, border=5, cuda=True, margin=0.2):
        super().__init__()
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.lam3 = float(lam3)
        self.margin = margin
        self.sample_size = sample_size
        self.sample_n = sample_n
        self.safe_radius_pos = safe_radius_pos
        self.safe_radius_neg = safe_radius_neg
        self.raw_grid = KU.create_meshgrid(input_size,input_size,normalized_coordinates=False).squeeze(0)
        self.AP = nn.AvgPool2d(sample_size+1,stride=1,padding=sample_size//2)
        self.MP = nn.MaxPool2d(sample_size+1,stride=1,padding=sample_size//2)
        self.MP3 = nn.MaxPool2d(3,stride=1,padding=1)
        self.MP7 = nn.MaxPool2d(7,stride=1,padding=3)
        self.AP3 = nn.AvgPool2d(3,stride=1,padding=1)
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2)
        self.border_mask = F.pad(torch.ones([input_size-2*border,input_size-2*border]),
                                 [border,border,border,border]).unsqueeze(0).unsqueeze(0).long()
        self.running_score_sum = 1000
        self.M_mean = 1000
        self.priori1_mean = 1000
        self.priori2_mean = 1000
        self.mask1_mean = 10
        self.mask2_mean = 10
        self.mask3_mean = 10
        self.running_rep_sum = 100
        self.loss_desc_ = 0
        self.loss_peak_ = 0
        self.loss_rep_ = 0
        self.loss_desc_align_ = 0
        if cuda:
            self.AP = self.AP.cuda()
            self.MP = self.MP.cuda()
            self.MP7 = self.MP7.cuda()
            self.MP3 = self.MP3.cuda()
            self.AP3 = self.AP3.cuda()
            self.AP5 = self.AP5.cuda()
            self.raw_grid = self.raw_grid.cuda()
            #self.b
            self.border_mask = self.border_mask.cuda().long()
    def ramdom_sampler(self,feat1,score1,feat2_warped,score2_warped,good_mask):
        mask = torch.rand_like(score1*1.0)*good_mask
        t = mask.view(-1)
        thres_new = t.topk(self.sample_n,largest=True)[0][-1]
        mask = mask >= thres_new
        mask = mask.squeeze(0).detach()
        feat1 = feat1[:,mask].t()
        feat2 = feat2_warped[:,mask].t()
        position = self.raw_grid[mask]
        score1 = score1[:,mask]
        score2 = score2_warped[:,mask]
        return feat1,score1,feat2,score2,position
    def compute_hard_dist(self,feat1,feat2,position):
        distx = (position[:,0].unsqueeze(1)-position[:,0].unsqueeze(0)).pow(2)
        disty = (position[:,1].unsqueeze(1)-position[:,1].unsqueeze(0)).pow(2)
        dist2 = distx + disty
        save_mask_neg = dist2<self.safe_radius_neg**2
        save_mask_pos = dist2<self.safe_radius_pos**2
        # hard mining
        simi = feat1 @ feat2.t()
        simi_a = feat1 @ feat1.t()
        simi_p = feat2 @ feat2.t()
        simi_max_pos = simi-10*(1-1.0*save_mask_pos)
        pos = torch.max(simi_max_pos,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi = simi - 10*(simi>0.9) - 10*save_mask_neg
        neg_n = torch.max(simi,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        neg_m = torch.max(simi,dim=1)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi_a = simi_a - 10*(simi_a>0.9) - 10*save_mask_neg
        neg_k = torch.max(simi_a,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi_p = simi_p - 10*(simi_p>0.9)  - 10*save_mask_neg
        neg_j = torch.max(simi_p,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        neg_cross = torch.max(neg_n,neg_m)
        M = ((torch.pi-neg_k.acos()).pow(2)/3+(torch.pi-neg_j.acos()).pow(2)/3+(torch.pi-neg_cross.acos()).pow(2)/3+\
            pos.acos().pow(2)).pow(2)
        return M

    def loss_desc_align(self):
        B, C, H, W = self.feat1.shape
        total_loss = 0
        count = 0

        raw_grid = self.raw_grid.reshape(-1, 2).float().to(self.feat1.device)

        feat2_unfold_all = self.feat2_warped.permute(0, 2, 3, 1).reshape(B, -1, C).detach()  

        for b in range(B):

            feat1_s, score1_s, feat2_s, score2_s, pos_grid = self.ramdom_sampler(
                self.feat1[b], self.AP3(self.score1[b]),
                self.feat2_warped[b], self.AP3(self.score2_warped[b]),
                self.good_mask[b]
            )

            N = feat1_s.size(0)
            if N < 10:
                continue

            feat1_s = F.normalize(feat1_s, dim=1)
            feat2_s = F.normalize(feat2_s, dim=1)

            sim_pos = (feat1_s * feat2_s).sum(dim=1)
            loss_pos = (1 - sim_pos).mean()

            patch1 = F.unfold(self.feat1[b:b+1], kernel_size=3, padding=1).squeeze(0).T  
            patch2 = F.unfold(self.feat2_warped[b:b+1], kernel_size=3, padding=1).squeeze(0).T
            flat_idx = (pos_grid[:, 1] * W + pos_grid[:, 0]).long() 
            p1 = F.normalize(patch1[flat_idx], dim=1)
            p2 = F.normalize(patch2[flat_idx], dim=1)
            loss_patch = (1 - (p1 * p2).sum(dim=1)).mean()

            with torch.no_grad():
                dists = torch.cdist(pos_grid.float(), raw_grid, p=2) 
                far_mask = dists > 5  

            feat2_neg = feat2_unfold_all[b]  
            feat2_neg = F.normalize(feat2_neg, dim=1)
            sim_neg_all = torch.matmul(feat1_s, feat2_neg.T) 
            sim_neg_all = sim_neg_all.masked_fill(~far_mask.to(sim_neg_all.device), float('-inf'))
            hard_neg = sim_neg_all.max(dim=1)[0]
            hard_filter = hard_neg > 0.1
            loss_neg = F.relu(hard_neg[hard_filter] - self.margin).mean() if hard_filter.any() else sim_neg_all.new_tensor(0.0)

            rand_indices = torch.randint(0, feat2_neg.size(0), (128,), device=feat2_neg.device)
            rand_neg = feat2_neg[rand_indices]
            rand_sim = torch.matmul(feat1_s, rand_neg.T)
            loss_rand = F.relu(rand_sim.mean(dim=1) - self.margin).mean()

            repul_indices = torch.randperm(feat2_neg.size(0), device=feat2_neg.device)[:128]
            repul_feat = feat2_neg[repul_indices]
            sim_repul = torch.matmul(repul_feat, repul_feat.T)
            repulsion_mask = ~torch.eye(128, device=sim_repul.device).bool()
            loss_repel = F.relu(sim_repul[repulsion_mask] - 0.8).mean()

            total_loss += loss_pos + 0.5 * loss_patch + 0.5 * loss_neg + 0.5 * loss_rand + 0.05 * loss_repel
            count += 1

        return total_loss / max(count, 1)

    def loss_desc(self):
        loss = 0
        for i in range(self.score1.shape[0]):
            
            feat1,score1,feat2,score2,position = self.ramdom_sampler(self.feat1[i],self.AP3(self.score1[i]),self.feat2_warped[i],
                                                                     self.AP3(self.score2_warped[i]),self.good_mask[i])
            assert feat1.shape[0]==feat2.shape[0]
            M = self.compute_hard_dist(feat1,feat2,position)
            score_i = score1*score2
            # generate save mask for two neighbors
            loss += (score_i.detach()*M).sum()/(self.running_score_sum+1e-5)
            self.running_score_sum = 0.99*self.running_score_sum+0.01*score_i.sum().detach()
            assert not loss.isnan()
        return loss/(i+1)
    
    def loss_rep(self):
        feat_wise_point_simi = ((self.feat1*self.feat2_warped).sum(dim=1,keepdim=True)).detach()*self.good_mask
        good_mask = F.unfold(feat_wise_point_simi,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches1 = F.unfold(self.score1*self.good_mask,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches1 = F.normalize(patches1,dim=2)
        patches2 = F.unfold(self.score2_warped*self.good_mask,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches2 = F.normalize(patches2,dim=2)
        patches_simi = F.unfold(feat_wise_point_simi,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches_simi = patches_simi.sum(dim=2,keepdim=True).clamp(min=0)/(good_mask.sum(dim=2,keepdim=True)+1)
        cosim = (patches1 * patches2).sum(dim=2,keepdim=True)
        #rep loss weighted with desciptors similairty
        loss_rep = (patches_simi*(1.0-cosim)).sum()/(self.running_rep_sum+1e-5)
        assert not loss_rep.isnan()
        self.running_rep_sum = 0.99*self.running_rep_sum + 0.01*patches_simi.sum()
        return loss_rep
    
    def compute_multiscale_sobel_softmax(self, image):

        sigma_base = 2.0
        ratio = 2 ** (1 / 3)
        Mmax = 8
        B, C, H, W = image.shape
        responses = []

        scale_values = torch.tensor(
            [sigma_base * (ratio ** i) for i in range(Mmax)],
            device=image.device
        )
        weights = (scale_values / scale_values.sum()).view(1, Mmax, 1, 1, 1)

        for i in range(Mmax):
            scale = scale_values[i].item()
            radius = int(round(2 * scale))
            ksize = 2 * radius + 1

            j = torch.arange(-radius, radius + 1, device=image.device).view(-1, 1)
            k = torch.arange(-radius, radius + 1, device=image.device).view(1, -1)
            xarry = k.repeat(ksize, 1)
            yarry = j.repeat(1, ksize)

            W = torch.exp(-(xarry**2 + yarry**2) / (2 * scale))

            W2 = torch.zeros_like(W)
            W2[radius + 1:, :] = +W[radius + 1:, :]
            W2[:radius, :] = -W[:radius, :]

            W1 = torch.zeros_like(W)
            W1[:, radius + 1:] = +W[:, radius + 1:]
            W1[:, :radius] = -W[:, :radius]

            kx = W1.unsqueeze(0).unsqueeze(0)
            ky = W2.unsqueeze(0).unsqueeze(0)

            Gx = F.conv2d(image, kx, padding=radius)
            Gy = F.conv2d(image, ky, padding=radius)

            grad = torch.sqrt(Gx**2 + Gy**2)
            grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            responses.append(grad)

        stack = torch.stack(responses, dim=1)  
        fused = (stack * weights).sum(dim=1)  

        fused = (fused - fused.amin(dim=(-2, -1), keepdim=True)) / (fused.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return fused

    def compute_multiscale_roewa_softmax(self, image):

        sigma_base = 2.0
        ratio = 2 ** (1 / 3)
        Mmax = 8
        B, C, H, W = image.shape
        responses = []

        scale_values = torch.tensor(
            [sigma_base * (ratio ** i) for i in range(Mmax)],
            device=image.device
        )
        weights = (scale_values / scale_values.sum()).view(1, Mmax, 1, 1, 1)

        for i in range(Mmax):
            scale = scale_values[i].item()
            radius = int(round(2 * scale))
            ksize = 2 * radius + 1

            j = torch.arange(-radius, radius + 1, device=image.device).view(-1, 1)
            k = torch.arange(-radius, radius + 1, device=image.device).view(1, -1)
            xarry = k.repeat(ksize, 1)
            yarry = j.repeat(1, ksize)

            W = torch.exp(-(torch.abs(xarry) + torch.abs(yarry)) / scale)

            W34 = torch.zeros_like(W)
            W12 = torch.zeros_like(W)
            W14 = torch.zeros_like(W)
            W23 = torch.zeros_like(W)

            W34[radius + 1:, :] = W[radius + 1:, :]
            W12[:radius, :] = W[:radius, :]
            W14[:, radius + 1:] = W[:, radius + 1:]
            W23[:, :radius] = W[:, :radius]

            k_M34 = W34.unsqueeze(0).unsqueeze(0)
            k_M12 = W12.unsqueeze(0).unsqueeze(0)
            k_M14 = W14.unsqueeze(0).unsqueeze(0)
            k_M23 = W23.unsqueeze(0).unsqueeze(0)

            M34 = F.conv2d(image, k_M34, padding=radius)
            M12 = F.conv2d(image, k_M12, padding=radius)
            M14 = F.conv2d(image, k_M14, padding=radius)
            M23 = F.conv2d(image, k_M23, padding=radius)

            ratio_Gx = (M14 + 1e-6) / (M23 + 1e-6)
            ratio_Gx = torch.clamp(ratio_Gx, min=1e-3, max=1e3)
            Gx = torch.log(ratio_Gx)

            ratio_Gy = (M34 + 1e-6) / (M12 + 1e-6)
            ratio_Gy = torch.clamp(ratio_Gy, min=1e-3, max=1e3)
            Gy = torch.log(ratio_Gy)

            grad = torch.sqrt(Gx**2 + Gy**2)
            grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            responses.append(grad)

        stack = torch.stack(responses, dim=1) 
        fused = (stack * weights).sum(dim=1)  

        fused = (fused - fused.amin(dim=(-2, -1), keepdim=True)) / (fused.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return fused

    def compute_edge(self, im, modality='optical'):
        
        if im.shape[1] == 3:
                im = im[:,0:1,:,:]

        if modality == 'optical':
            edge = self.compute_multiscale_sobel_softmax(im)
        elif modality == 'sar':
            edge = self.compute_multiscale_roewa_softmax(im)

        return edge.detach()

    def loss_peak(self):
        priori1 = self.compute_edge(self.im1, modality='optical')
        mask1 = 1-priori1/(priori1.mean()+1e-12)
        #mask_pos = mask
        mask1 = F.relu(mask1)
        priori2 = self.compute_edge(self.im2, modality='sar')
        mask2 = 1-priori2/(priori2.mean()+1e-12)
        mask2 = F.relu(mask2)
        score1 = self.score1*self.border_mask
        score2 = self.score2*self.border_mask
        score1_ = KF.gaussian_blur2d(score1,kernel_size=(3,3),sigma=(1,1))
        score2_ = KF.gaussian_blur2d(score2,kernel_size=(3,3),sigma=(1,1))
        loss_peak_edge = (mask1*score1.pow(2)).mean()/self.mask1_mean + (mask2*score2.pow(2)).mean()/self.mask2_mean

        loss_peak_random = self.AP(score1_).pow(2).mean() + (1-self.MP(score1_)).pow(2).mean() + (self.AP3(score1_)+1-self.MP3(score1_)).pow(2).mean()+\
                           self.AP(score2_).pow(2).mean() + (1-self.MP(score2_)).pow(2).mean() + (self.AP3(score2_)+1-self.MP3(score2_)).pow(2).mean()

        loss_peak_coupled = 0
        for i in range(self.score1.shape[0]):
            feat1,score1,feat2,score2,position = self.ramdom_sampler(self.feat1[i],self.score1[i],self.feat2_warped[i],
                                                                     self.score2_warped[i],self.good_mask[i])
            assert feat1.shape[0]==feat2.shape[0]
            M = self.compute_hard_dist(feat1,feat2,position)
            M = (M-M.min())/(M.max()-M.min())
            mask3 = F.relu((1-M/(M.mean()+1e-12)))
            mask3 = mask3.detach()
            
            t = (mask3*(1-score1).pow(2)).mean()+(mask3*(1-score2).pow(2)).mean()
            loss_peak_coupled += t/self.mask3_mean
            self.mask3_mean = 0.99*self.mask3_mean + 0.01*mask3.mean()
        loss_peak_coupled = loss_peak_coupled/(i+1)
        self.mask1_mean = 0.99*self.mask1_mean + 0.01*mask1.mean()
        self.mask2_mean = 0.99*self.mask2_mean + 0.01*mask2.mean()
        loss_peak = (loss_peak_edge+loss_peak_random+loss_peak_coupled)
        assert not loss_peak_edge.isnan()
        assert not loss_peak_coupled.isnan()
        assert not loss_peak_random.isnan()
        assert not loss_peak.isnan()
        return loss_peak
    
    def forward(self,feat1,score1,feat2,score2,flow12=None,img1=None,img2=None):
        ones = torch.ones_like(score1)
        score1 = score1
        score2 = score2
        self.good_mask = self.generate_good_mask(ones,flow12)*self.border_mask # filter border
        self.feat2_warped = KGT.remap(feat2,flow12[..., 0], flow12[..., 1], align_corners=False,mode='nearest')
        self.score2_warped = KGT.remap(score2,flow12[..., 0], flow12[..., 1], align_corners=False,mode='bilinear')
        self.feat1 = feat1
        self.feat2 = feat2
        self.score1 = score1
        self.score2 = score2
        self.flow12 = flow12
        self.im1 = img1
        self.im2 = img2
        self.loss_desc_ = self.loss_desc()
        self.loss_peak_ = self.loss_peak()
        self.loss_rep_ = self.loss_rep()
        self.loss_desc_align_ = self.loss_desc_align()
        
        return self.loss_desc_+self.lam1*self.loss_peak_ + self.lam2*self.loss_rep_ + self.lam3 * self.loss_desc_align_
    
    def generate_good_mask(self,ones,flow12,n=1):
        good_mask =KGT.remap(ones*0+1,flow12[..., 0], flow12[..., 1], align_corners=False)
        t = flow12
        for i in range(n):
            good_mask = self.AP3(good_mask.float())>0.5
        return good_mask
        

