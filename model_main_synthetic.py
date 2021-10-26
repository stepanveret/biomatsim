"""
Code is based on: https://github.com/AMLab-Amsterdam/DIVA
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from diva.pixel_cnn_utils import log_mix_dep_Logistic_256
from diva.resnet_blocks_batchnorm import *
from model_vae_synthetic import Encoder, Decoder, NormalModule
from model_qzfreg_synthetic import pzf, qf


# Decoder (concatenated latent space)
class px(Decoder):
    def __init__(self, zeps_dim, zf_dims):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        """
        super(px, self).__init__(zeps_dim)
        self.fc1 = nn.Sequential(nn.Linear(sum(zf_dims) + zeps_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)

    def forward(self, zeps, zfs):
        """
        zfs - list of tensors with shapes: [(batch, 64), (batch, 64), ...]
        """
        zepszfs = torch.cat((zeps, torch.cat(zfs, dim=1)), dim=1)

        return super(px, self).forward(zepszfs)


# class px1(px):  # p(x|ze, zf)
#     def __init__(self, zeps_dim, zf_dims):
#         """
#         zf_dims = [zf1_dim, zf2_dim, ...]
#         """
#         super(px1, self).__init__(zeps_dim, zf_dims)
#         del fc1
#         # extra input dimension
#         self.fc1 = nn.Sequential(nn.Linear(sum(zf_dims) + zeps_dim + 1, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
#         torch.nn.init.xavier_uniform_(self.fc1[0].weight)
#         self.fc1[0].bias.data.zero_()


class FDVAE(nn.Module):
    """
    Cell model with latent space disentangled by features.
    """
    def __init__(self, args):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        f_dims = [f1_dim, f2_dim, ...]
        """
        super(FDVAE, self).__init__()
        self.num_features = len(args.zf_dims)
        self.zf_dims = args.zf_dims
        self.zeps_dim = args.zeps_dim
        self.f_dims = args.f_dims
        self.eps_dim = args.eps_dim
        self.qf_activations = args.qf_activations

        # self.splitdim = args.splitdim
        # if self.splitdim:
        #     self.mult_loc = args.mult_loc
        #     self.mult_scale = args.mult_scale
        #     self.px = px1(self.zeps_dim, self.zf_dims)
        # else:
        self.px = px(self.zeps_dim, self.zf_dims)  # p(x|ze, zf)

        self.pzf_list = nn.ModuleList()  # [p(zf1|f1), p(zf2|f2), ...]
        for f_dim, zf_dim in zip(self.f_dims, self.zf_dims):
            pzf_k = pzf(f_dim, zf_dim)
            self.pzf_list.append(pzf_k)

        self.qzeps = Encoder(self.zeps_dim)  # q(ze|x)
        self.qzf_list = nn.ModuleList()  # [q(zf1|x), q(zf2|x), ...]
        for zf_dim in self.zf_dims:
            qzf_k = Encoder(zf_dim)
            self.qzf_list.append(qzf_k)

        self.qf_list = nn.ModuleList()  # [q(f1|zf1), q(f2|zf2), ...]
        for f_dim, zf_dim, qf_act_bool in zip(self.f_dims, self.zf_dims, self.qf_activations):
            qf_k = qf(f_dim, zf_dim, qf_act_bool)
            self.qf_list.append(qf_k)

        self.aux_loss_multipliers = args.aux_loss_multipliers  # [alpha_f1, alpha_f2, ...]
        self.aux_loss_multipliers_2 = args.aux_loss_multipliers_2  # [alpha_f1, alpha_f2, ...] (= 0, not used)

        self.betas_f = args.betas_f  # [beta_f1, beta_f2, ...] for KL(q(zf|x) || p(zf|f))
        self.beta_eps = args.beta_eps  # for KL(q(ze|x) || p(ze))

        ### https://arxiv.org/abs/1804.03599 approach for z_eps
        self.gamma = args.gamma
        self.c = args.c
        ###

        self.pzf_full_prior = args.pzf_full_prior
        self.betas_f_full = args.betas_f_full # [beta_f1_full, beta_f2_full, ...] for KL(p(zf|f) || p(zf))

        self.prior_zeps_scale_train = args.prior_zeps_scale_train
        self.prior_zeps_log_scale = nn.Parameter(torch.Tensor([0.0] * int(self.zeps_dim)))
        self.prior_zf_full_scale_train_list = args.prior_zf_full_scale_train_list
        self.prior_zf_full_log_scales = nn.ParameterList()

        for k in range(self.num_features):
            self.prior_zf_full_log_scales.append(nn.Parameter(torch.Tensor([0.0] * int(self.zf_dims[k]))))

        # self.pi = torch.acos(torch.zeros(1)).item() * 2
        # self.e = torch.exp(torch.tensor(1.)).item()

        self.cuda()

    # def smoothdim(self, tensor):
    #     return (self.pi * torch.sigmoid(tensor) / 2 - self.pi / 4) * torch.log(torch.abs(tensor) + self.e)

    # def smoothdim_tanh(self, tensor, mult):
    #     return torch.tanh(tensor) * mult * torch.log(torch.sqrt(tensor * tensor.clone().detach()) + self.e)

    def forward(self, x, fs):
        """
        x = (batch_size, 3, 128, 128)
        fs = [(batch_size, 1), (batch_size, N), ...]; fs[k] array corresponds to fk
        """
        ##### Encode; get parameters
        ## ze
        zeps_q_loc, zeps_q_scale = self.qzeps(x)  # q(ze|x_i) -> qze_loc, qze_scale

        ## zf
        zf_q_locs = []
        zf_q_scales = []
        for k in range(self.num_features):
            zfk_q_loc, zfk_q_scale = self.qzf_list[k](x)
            zf_q_locs.append(zfk_q_loc)
            zf_q_scales.append(zfk_q_scale)

        # if self.splitdim:
        #     zeps_q_loc[:, 0] = self.smoothdim_tanh(zeps_q_loc[:, 0], mult=self.mult_loc)
        #     zeps_q_scale[:, 0] = self.smoothdim_tanh(zeps_q_scale[:, 0], mult=self.mult_scale)  # mult=self.pi / 16

        ##### Reparameterization trick (encoders)
        ## ze
        qzeps = dist.Normal(zeps_q_loc, zeps_q_scale)  # Normal distribution q(ze|x_i)
        zeps_q = qzeps.rsample()  # sample ze_j ~ q(ze|x_i)

        ## zf
        qzf_list = nn.ModuleList()  # Normal distributions [q(zf1|x_i), q(zf2|x_i), ...]
        zf_q_list = []  # samples [zf1_j, zf2_j, ...], where zfk_j ~ q(zfk|x_i)
        for k in range(self.num_features):
            qzf_k = NormalModule(zf_q_locs[k], zf_q_scales[k])
            qzf_k.do_rsample()  # sample zfk_j ~ q(zfk|x_i) Normal
            zfk_q = qzf_k.get_rsample()
            qzf_list.append(qzf_k)
            zf_q_list.append(zfk_q)  # only for x_recon

        ##### Decode
        # if self.splitdim:
        #     zeps_q_0_sin2 = torch.sin(2 * zeps_q[:, 0]).reshape(-1, 1)
        #     zeps_q_0_cos = torch.abs(torch.cos(zeps_q[:, 0])).reshape(-1, 1)
        #     zeps_q_extended = torch.cat((zeps_q_0_sin2, zeps_q_0_cos, zeps_q[:, 1:]), dim=1)
        #     x_recon = self.px(zeps_q_extended, zf_q_list)
        # else:
        x_recon = self.px(zeps_q, zf_q_list)  # x_i_recon ~ p(x|ze_j, [zf1_j, zf2_j, ...])

        ##### Get parameters for priors
        ## zf
        zf_p_locs = []
        zf_p_scales = []
        zf_p_full_locs = []
        zf_p_full_scales = []
        for k in range(self.num_features):
            """
            fs[k].shape: (batch_size, 1) or (batch_size, N) for a cat. feature with N values
            """
            zfk_p_loc, zfk_p_scale = self.pzf_list[k](fs[k].float())  # p(zfk|fk(x_i))
            zf_p_locs.append(zfk_p_loc)
            zf_p_scales.append(zfk_p_scale)

            #### full priors; zfk_p_loc.size()[0] serves as a batch size
            zfk_p_full_loc = torch.zeros(zfk_p_loc.size()[0], self.zf_dims[k]).cuda()
            if self.prior_zf_full_scale_train_list[k]:
                zfk_p_full_log_scale = F.normalize(torch.sigmoid(self.prior_zf_full_log_scales[k]), p=2, dim=0)
                zfk_p_full_scale = zfk_p_full_log_scale.reshape(1, -1).repeat(zfk_p_loc.size()[0], 1)
            else:
                zfk_p_full_scale = torch.ones(zfk_p_loc.size()[0], self.zf_dims[k]).cuda()            

            zf_p_full_locs.append(zfk_p_full_loc)
            zf_p_full_scales.append(zfk_p_full_scale)
            ####

        ## ze
        zeps_p_loc = torch.zeros(zfk_p_loc.size()[0], self.zeps_dim).cuda() # prior p(ze) -> pze_loc, pze_scale
        if self.prior_zeps_scale_train:
            zeps_p_log_scale = F.normalize(torch.sigmoid(self.prior_zeps_log_scale), p=2, dim=0)
            zeps_p_scale = zeps_p_log_scale.reshape(1, -1).repeat(zfk_p_loc.size()[0], 1)
        else:
            zeps_p_scale = torch.ones(zfk_p_loc.size()[0], self.zeps_dim).cuda()

        ##### Reparameterization trick (priors)
        ## ze
        pzeps = dist.Normal(zeps_p_loc, zeps_p_scale)  # Normal distribution p(ze) - prior

        ## zf
        pzf_list = nn.ModuleList()  # Normal distribtuions [p(zf1|f1(x_i)), p(zf2|f2(x_i)), ...] - cond. priors
        zf_p_list = [] # samples [zf1_s, zf2_s, ...], where zfk_s ~ p(zfk|fk(x_i))
        pzf_full_list = nn.ModuleList()  # Normal distribtuions [p(zf1), p(zf2), ...] - priors without f
        for k in range(self.num_features):
            pzf_k = NormalModule(zf_p_locs[k], zf_p_scales[k])
            #### 
            pzf_k.do_rsample()  # sample zfk_s ~ p(zfk|fk(x_i)) Normal (for full prior)
            zfk_p = pzf_k.get_rsample()
            pzf_list.append(pzf_k)
            zf_p_list.append(zfk_p)  # this list is not used
            pzf_k_full = NormalModule(zf_p_full_locs[k], zf_p_full_scales[k])
            pzf_full_list.append(pzf_k_full)
            ####

        ##### Auxiliary losses (predictions based on samples from posteriors)
        f_hat_list = []  # [q(f1|zf1_j), q(f2|zf2_j), ...] -> [f1_hat, f2_hat, ...]
        for k in range(self.num_features):
            qfk = self.qf_list[k]  # q(fk|zfk)
            zfk_q = zf_q_list[k]  # zfk_j ~ q(zfk|x_i)
            fk_hat = qfk(zfk_q)
            f_hat_list.append(fk_hat)

        ##### Auxiliary losses 2 (predictions based on samples from conditional priors - not used for training)
        f_hat_list2 = []  # [q(f1|zf1_s), q(f2|zfk_s), ...] -> [f1_hat2, f2_hat2, ...]
        for k in range(self.num_features):
            qfk = self.qf_list[k]  # q(fk|zfk)
            zfk_p = zf_p_list[k]  # zfk_s ~ p(zfk|fk(x_i)) Normal
            fk_hat2 = qfk(zfk_p)
            f_hat_list2.append(fk_hat2)

        return x_recon, f_hat_list, f_hat_list2, qzf_list, pzf_list, pzf_full_list, zf_q_list, zf_p_list, qzeps, pzeps, zeps_q

    def loss_function(self, x, fs):
        """
        Step 1 of the training procedure.

        x = (batch_size, 3, 128, 128)
        fs = [(batch_size, 1), (batch_size, N), ...]; fs[k] array corresponds to fk
        """
        x_recon, f_hat_list, f_hat_list2, qzf_list, pzf_list, pzf_full_list,\
                        zf_q_list, zf_p_list, qzeps, pzeps, zeps_q = self.forward(x, fs)

        CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)  # minimize
        max_objective = -CE_x  # ~~ log-likelihood of the data
           
        KL_zeps = torch.sum(qzeps.log_prob(zeps_q) - pzeps.log_prob(zeps_q))
            # pzeps: Normal distribution N(0, I)
            # qzeps: Normal distribution q(ze|x_i); self.qzeps(x) -> N(loc(x), scale(x))
            # zeps_q: sample ze_j ~ q(ze|x_i)
            # Given an observation x: KL(q(ze|x)||p(ze)) = E[log q(ze|x) - log p(ze)], ze ~  q(ze|x);
            # Finally, sum by x (batch)

        if self.gamma > 0:  # 
            max_objective -= self.gamma * torch.abs(KL_zeps - self.c)
        else:
            max_objective -= self.beta_eps * KL_zeps

        MSE_fs = []
        MSE_fs_2 = []
        KLs_zfk = []
        KLs_zfk_full = []
        for k in range(self.num_features):
            pzf_k = pzf_list[k]  # Normal distribution p(zfk|fk(x_i)) - conditional prior
            qzf_k = qzf_list[k]  # Normal distribution q(zfk|x_i) - encoder
            zfk_q = qzf_k.get_rsample()  # zfk_j ~ q(zfk|x_i)
            KL_zfk = torch.sum(qzf_k.normal.log_prob(zfk_q) - pzf_k.normal.log_prob(zfk_q))
            max_objective -= self.betas_f[k] * KL_zfk
            KLs_zfk.append(KL_zfk)
    
            # conditional prior regularization
            pzf_k_full = pzf_full_list[k]  # p(zfk|fk(x_i)) - full prior
            zfk_p = pzf_k.get_rsample()  # zfk_s ~ p(zfk|fk(x_i))
            KL_zfk_full = torch.sum(pzf_k.normal.log_prob(zfk_p) - pzf_k_full.normal.log_prob(zfk_p))
            KLs_zfk_full.append(KL_zfk_full)
            if self.pzf_full_prior:
                max_objective -= self.betas_f_full[k] * KL_zfk_full

            # aux. objective
            fk_prediction = f_hat_list[k].float()
            fk_target = fs[k].float()
            MSE_k = F.mse_loss(fk_prediction, fk_target)  # .cross_entropy(fk_prediction, fk_target, reduction='sum')
            MSE_fs.append(MSE_k)
            max_objective -= self.aux_loss_multipliers[k] * MSE_k

            # aux. objective 2
            fk_prediction2 = f_hat_list2[k].float()
            MSE_k_2 = F.mse_loss(fk_prediction2, fk_target)
            MSE_fs_2.append(MSE_k_2)
            max_objective -= self.aux_loss_multipliers_2[k] * MSE_k_2

        loss = -max_objective

        return loss, CE_x, KL_zeps, KLs_zfk, KLs_zfk_full, MSE_fs, MSE_fs_2


class pl(Decoder):  # p(x|ze, zf, le)
    """
    Topography decoder for the case when zeps is also used as part of the topography latent space.
    """
    def __init__(self, zeps_dim, zf_dims, leps_dim):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        """
        super(pl, self).__init__(leps_dim)
        self.fc1 = nn.Sequential(nn.Linear(sum(zf_dims) + zeps_dim + leps_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)

    def forward(self, zeps, zfs, leps):
        """
        zfs - list of tensors with shapes: [(batch, 2), (batch, 2), ...]
        """
        zepszfsleps = torch.cat((zeps, torch.cat(zfs, dim=1), leps), dim=1)

        return super(pl, self).forward(zepszfsleps)


class pl1(Decoder):  # p(x|zf, le)
    """
    Topography decoder (only zf(s) are used as part of the topography latent space).
    """
    def __init__(self, zf_dims, leps_dim):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        """
        super(pl1, self).__init__(leps_dim)
        self.fc1 = nn.Sequential(nn.Linear(sum(zf_dims) + leps_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)

    def forward(self, zfs, leps):
        """
        zfs - list of tensors with shapes: [(batch, 2), (batch, 2), ...]
        """
        zfsleps = torch.cat((torch.cat(zfs, dim=1), leps), dim=1)

        return super(pl1, self).forward(zfsleps)


class REGVAE(nn.Module):
    """
    Full model.
    """
    def __init__(self, args):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        f_dims = [f1_dim, f2_dim, ...]
        """
        super(REGVAE, self).__init__()
        self.fdvae = args.fdvae  # cell model
        self.use_zeps = args.use_zeps  # if we'd like to use zeps as part of topography latent space

        self.leps_dim = args.leps_dim
        if self.use_zeps:
            self.pl = pl(self.fdvae.zeps_dim, self.fdvae.zf_dims, self.leps_dim)  # decoder
        else:
            self.pl = pl1(self.fdvae.zf_dims, self.leps_dim) 

        if self.use_zeps:
            self.qlzeps = Encoder(self.fdvae.zeps_dim)  # encoder ze
        self.qleps = Encoder(self.leps_dim)  # encoder le

        self.lqzf_list = nn.ModuleList()  # [q(zf1|x), q(zf2|x), ...]
        for zf_dim in self.fdvae.zf_dims:
            lqzf_k = Encoder(zf_dim)  # encoder(s) zf
            self.lqzf_list.append(lqzf_k)

        self.lbetas_f = args.lbetas_f  # [beta_f1, beta_f2, ...] for KL(q(zf|l) || p(zf)) (p(zf) - full prior)
        if self.use_zeps:
            self.lbeta_eps = args.lbeta_eps  # for KL(q(ze|l) || p(ze))
        self.beta_leps = args.beta_leps  # for KL(q(le|l) || p(le))

        ### https://arxiv.org/abs/1804.03599 approach for l_eps
        self.gamma_leps = args.gamma_leps
        self.c_leps = args.c_leps
        ###

        self.eta = args.eta  # for CE_l(l_recon, l) where zf ~ q(zf|x), ze ~ q(ze|x) (auxiliary objective 2B)

        self.prior_leps_scale_train = args.prior_leps_scale_train
        self.prior_leps_log_scale = nn.Parameter(torch.Tensor([0.0] * int(self.leps_dim)))

        ##### Not used for optimization
        self.beta_combined = args.beta_combined
        self.beta_combined2 = args.beta_combined2

        if self.use_zeps:
            self.combined_dim = int(self.fdvae.zeps_dim + sum(self.fdvae.zf_dims) + self.leps_dim)
        else:
            self.combined_dim = int(sum(self.fdvae.zf_dims) + self.leps_dim)

        self.kl_llh_multiplier = args.kl_llh_multiplier
        self.delta = args.delta  # for CE_l(x_recon, x) * (KL + delta), where ze ~ q(ze|l), zf ~ q(zf|l) (not used)
        #####

        self.cuda()

    def forward_l(self, l): 
        """
        Topography image reconstruction. 
        l -> zf/le -> l' (l -> ze/zf/le -> l')

        l = (batch_size, 3, 128, 128)
        """
        ##### Encode; get parameters
        ## ze
        if self.use_zeps:
            zeps_q_loc, zeps_q_scale = self.qlzeps(l)  # q(ze|l_i) -> qze_loc, qze_scale

        ## le
        leps_q_loc, leps_q_scale = self.qleps(l)   # q(le|l_i) -> qle_loc, qle_scale

        ## zf
        zf_q_locs = []
        zf_q_scales = []
        for k in range(self.fdvae.num_features):
            zfk_q_loc, zfk_q_scale = self.lqzf_list[k](l)
            zf_q_locs.append(zfk_q_loc)
            zf_q_scales.append(zfk_q_scale)

        ##### Combined posterior (not used for training)
        if self.use_zeps:
            combined_q_loc = torch.cat((zeps_q_loc, torch.cat(zf_q_locs, dim=1), leps_q_loc), dim=1)
            combined_q_scale = torch.cat((zeps_q_scale, torch.cat(zf_q_scales, dim=1), leps_q_scale), dim=1)      
        else:
            combined_q_loc = torch.cat((torch.cat(zf_q_locs, dim=1), leps_q_loc), dim=1)
            combined_q_scale = torch.cat((torch.cat(zf_q_scales, dim=1), leps_q_scale), dim=1)   

        ##### Reparameterization trick (encoders)
        ## ze
        if self.use_zeps:
            qzeps = dist.Normal(zeps_q_loc, zeps_q_scale)  # Normal distribution q(ze|l_i)
            zeps_q = qzeps.rsample()  # sample ze_j ~ q(ze|l_i)

        ## le
        qleps = dist.Normal(leps_q_loc, leps_q_scale)  # Normal distribution q(le|l_i)
        leps_q = qleps.rsample()  # sample le_j ~ q(le|l_i)

        ## zf
        qzf_list = nn.ModuleList()  # Normal distributions [q(zf1|l_i), q(zf2|l_i), ...]
        zf_q_list = []  # samples [zf1_j, zf2_j, ...], where zfk_j ~ q(zfk|l_i)
        for k in range(self.fdvae.num_features):
            qzf_k = NormalModule(zf_q_locs[k], zf_q_scales[k])
            qzf_k.do_rsample()  # sample zfk_j ~ q(zfk|x_i) Normal
            zfk_q = qzf_k.get_rsample()
            qzf_list.append(qzf_k)
            zf_q_list.append(zfk_q)  # only for l_recon

        ##### Reparameterization trick combined (not used for training)
        qcombined = dist.Normal(combined_q_loc, combined_q_scale)
        combined_q = qcombined.rsample()

        ##### Decode
        if self.use_zeps:
            l_recon = self.pl(zeps_q, zf_q_list, leps_q)  # l_i_recon ~ p(l|ze_j, [zf1_j, zf2_j, ...], le_j)
        else:
            l_recon = self.pl(zf_q_list, leps_q)

        ##### Get parameters for priors
        ## zf
        zf_p_locs = []
        zf_p_scales = []
        zf_p_log_scales_normalized = []
        for k in range(self.fdvae.num_features):
            zfk_p_loc = torch.zeros(leps_q_loc.size()[0], self.fdvae.zf_dims[k]).cuda()
            if self.fdvae.prior_zf_full_scale_train_list[k]:
                zfk_p_log_scale = F.normalize(torch.sigmoid(self.fdvae.prior_zf_full_log_scales[k]), p=2, dim=0)
                zf_p_log_scales_normalized.append(zfk_p_log_scale)
                zfk_p_scale = zfk_p_log_scale.reshape(1, -1).repeat(leps_q_loc.size()[0], 1)
            else:
                zfk_p_scale = torch.ones(leps_q_loc.size()[0], self.fdvae.zf_dims[k]).cuda()   

            zf_p_locs.append(zfk_p_loc)
            zf_p_scales.append(zfk_p_scale)

        ## ze
        if self.use_zeps:
            zeps_p_loc = torch.zeros(leps_q_loc.size()[0], self.fdvae.zeps_dim).cuda()
            if self.fdvae.prior_zeps_scale_train:
                zeps_p_log_scale = F.normalize(torch.sigmoid(self.fdvae.prior_zeps_log_scale), p=2, dim=0)
                zeps_p_scale = zeps_p_log_scale.reshape(1, -1).repeat(leps_q_loc.size()[0], 1)
            else:
                zeps_p_scale = torch.ones(leps_q_loc.size()[0], self.fdvae.zeps_dim).cuda()

        ## le
        leps_p_loc = torch.zeros(leps_q_loc.size()[0], self.leps_dim).cuda()
        if self.prior_leps_scale_train:
            leps_p_log_scale = F.normalize(torch.sigmoid(self.prior_leps_log_scale), p=2, dim=0)
            leps_p_scale = leps_p_log_scale.reshape(1, -1).repeat(leps_q_loc.size()[0], 1)
        else:
            leps_p_scale = torch.ones(leps_q_loc.size()[0], self.leps_dim).cuda()

        ##### Parameters for combined prior (not used for training)
        combined_p_loc = torch.zeros(leps_q_loc.size()[0], self.combined_dim).cuda()            
        if self.prior_leps_scale_train:
            if self.use_zeps: 
                combined_p_log_scale = torch.cat((zeps_p_log_scale,
                                                  torch.cat(zf_p_log_scales_normalized, dim=0),
                                                  leps_p_log_scale), dim=0)
            else: 
                combined_p_log_scale = torch.cat((torch.cat(zf_p_log_scales_normalized, dim=0),
                                                  leps_p_log_scale), dim=0)
            combined_p_scale = combined_p_log_scale.reshape(1, -1).repeat(leps_q_loc.size()[0], 1)
        else:
            combined_p_scale = torch.ones(leps_q_loc.size()[0], self.combined_dim).cuda()

        ##### Priors
        ## ze
        if self.use_zeps:
            pzeps = dist.Normal(zeps_p_loc, zeps_p_scale)  # Normal distribution p(ze) - prior

        ## le
        pleps = dist.Normal(leps_p_loc, leps_p_scale)  # Normal distribution p(le) - prior

        ## zf
        pzf_list = nn.ModuleList()  # Normal distribtuions [p(zf1), p(zf2), ...] - priors without f
        for k in range(self.fdvae.num_features):
            pzf_k = NormalModule(zf_p_locs[k], zf_p_scales[k])
            pzf_list.append(pzf_k)

        ##### Combined prior (not used for training)
        pcombined = dist.Normal(combined_p_loc, combined_p_scale)

        if self.use_zeps:
            return l_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q,\
                   qleps, pleps, leps_q, qcombined, pcombined, combined_q
        else:
            return l_recon, qzf_list, pzf_list, zf_q_list, None, None, None,\
                   qleps, pleps, leps_q, qcombined, pcombined, combined_q

    def forward_lx(self, l, x=None): 
        """
        Generate cell images based on a topography. 
        l -> zf/(le) + zeps -> x'  (l -> ze/zf/(le) -> x')
        """
        ##### Encode; get parameters
        ## ze
        if self.use_zeps:
            zeps_q_loc, zeps_q_scale = self.qlzeps(l)  # q(ze|l_i) -> qze_loc, qze_scale
        elif x is not None:
            zeps_q_loc, zeps_q_scale = self.fdvae.qzeps(x)  # training scheme 2Ax (default)
        else:  # sample from prior (training scheme 2A)
            zeps_q_loc = torch.zeros(l.size()[0], self.fdvae.zeps_dim).cuda()
            if self.fdvae.prior_zeps_scale_train:
                zeps_q_log_scale = F.normalize(torch.sigmoid(self.fdvae.prior_zeps_log_scale), p=2, dim=0)
                zeps_q_scale = zeps_q_log_scale.reshape(1, -1).repeat(l.size()[0], 1)
            else:
                zeps_q_scale = torch.ones(l.size()[0], self.fdvae.zeps_dim).cuda()

        ## zf
        zf_q_locs = []
        zf_q_scales = []
        for k in range(self.fdvae.num_features):
            zfk_q_loc, zfk_q_scale = self.lqzf_list[k](l)  # q(le|l_i) -> qle_loc, qle_scale
            zf_q_locs.append(zfk_q_loc)
            zf_q_scales.append(zfk_q_scale)

        ##### Reparameterization trick (encoders)
        ## ze
        qzeps = dist.Normal(zeps_q_loc, zeps_q_scale)  # Normal distribution q(ze|l_i)
        zeps_q = qzeps.rsample()  # sample ze_j ~ q(ze|l_i)

        ## zf
        qzf_list = nn.ModuleList()  # Normal distributions [q(zf1|l_i), q(zf2|l_i), ...]
        zf_q_list = []  # samples [zf1_j, zf2_j, ...], where zfk_j ~ q(zfk|l_i)
        for k in range(self.fdvae.num_features):
            qzf_k = NormalModule(zf_q_locs[k], zf_q_scales[k])
            qzf_k.do_rsample()  # sample zfk_j ~ q(zfk|x_i) Normal
            zfk_q = qzf_k.get_rsample()
            qzf_list.append(qzf_k)
            zf_q_list.append(zfk_q)  # only for l_recon

        ##### Decode
        x_recon = self.fdvae.px(zeps_q, zf_q_list)  # x_i_recon ~ p(x|ze_j, [zf1_j, zf2_j, ...])

        ##### Get parameters for priors
        ## zf
        zf_p_locs = []
        zf_p_scales = []
        for k in range(self.fdvae.num_features):
            zfk_p_loc = torch.zeros(zeps_q_loc.size()[0], self.fdvae.zf_dims[k]).cuda()
            if self.fdvae.prior_zf_full_scale_train_list[k]:
                zfk_p_log_scale = F.normalize(torch.sigmoid(self.fdvae.prior_zf_full_log_scales[k]), p=2, dim=0)
                zfk_p_scale = zfk_p_log_scale.reshape(1, -1).repeat(zeps_q_loc.size()[0], 1)
            else:
                zfk_p_scale = torch.ones(zeps_q_loc.size()[0], self.fdvae.zf_dims[k]).cuda()  

            zf_p_locs.append(zfk_p_loc)
            zf_p_scales.append(zfk_p_scale)

        ## ze
        if self.use_zeps:
            zeps_p_loc = torch.zeros(zeps_q_loc.size()[0], self.fdvae.zeps_dim).cuda()
            if self.fdvae.prior_zeps_scale_train:
                zeps_p_log_scale = F.normalize(torch.sigmoid(self.fdvae.prior_zeps_log_scale), p=2, dim=0)
                zeps_p_scale = zeps_p_log_scale.reshape(1, -1).repeat(zeps_q_loc.size()[0], 1)
            else:
                zeps_p_scale = torch.ones(zeps_q_loc.size()[0], self.fdvae.zeps_dim).cuda()

        ##### Priors
        ## ze
        if self.use_zeps:
            pzeps = dist.Normal(zeps_p_loc, zeps_p_scale)  # Normal distribution p(ze) - prior

        ## zf
        pzf_list = nn.ModuleList()  # Normal distribtuions [p(zf1), p(zf2), ...] - priors without f
        for k in range(self.fdvae.num_features):
            pzf_k = NormalModule(zf_p_locs[k], zf_p_scales[k])
            pzf_list.append(pzf_k)

        if self.use_zeps:
            return x_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q, None, None, None
                #  l_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q, qleps, pleps, leps_q
        else:
            return x_recon, qzf_list, pzf_list, zf_q_list, qzeps, None, zeps_q, None, None, None

    def forward_xl(self, l, x): 
        """
        Generate topographies based on a cell image. Training sceme 2Bp: infer le based on l (not used).
        x, l -> zf, le -> l' (x, l -> ze/zf, le -> l')
        """
        ##### Encode; get parameters
        ## le
        leps_q_loc, leps_q_scale = self.qleps(l)  # !

        ################################# based on a cell image #############################
        ## ze
        if self.use_zeps:
            zeps_q_loc, zeps_q_scale = self.fdvae.qzeps(x)  # q(ze|x_i) -> qze_loc, qze_scale

        ## zf
        zf_q_locs = []
        zf_q_scales = []
        for k in range(self.fdvae.num_features):
            zfk_q_loc, zfk_q_scale = self.fdvae.qzf_list[k](x)
            zf_q_locs.append(zfk_q_loc)
            zf_q_scales.append(zfk_q_scale)

        ##### Reparameterization trick (encoders)
        ## ze
        if self.use_zeps:
            qzeps = dist.Normal(zeps_q_loc, zeps_q_scale)  # Normal distribution q(ze|x_i)
            zeps_q = qzeps.rsample()  # sample ze_j ~ q(ze|x_i)

        ## zf
        qzf_list = nn.ModuleList()  # Normal distributions [q(zf1|x_i), q(zf2|x_i), ...]
        zf_q_list = []  # samples [zf1_j, zf2_j, ...], where zfk_j ~ q(zfk|x_i)
        for k in range(self.fdvae.num_features):
            qzf_k = NormalModule(zf_q_locs[k], zf_q_scales[k])
            qzf_k.do_rsample()  # sample zfk_j ~ q(zfk|x_i) Normal
            zfk_q = qzf_k.get_rsample()
            qzf_list.append(qzf_k)
            zf_q_list.append(zfk_q)  # only for x_recon
        ########################################################################################

        ## le
        qleps = dist.Normal(leps_q_loc, leps_q_scale)  # Normal distribution q(le|l_i)
        leps_q = qleps.rsample()  # sample le_j ~ q(le|l_i)

        ##### Decode
        if self.use_zeps:
            l_recon = self.pl(zeps_q, zf_q_list, leps_q)  # l_i_recon ~ p(l|ze_j, [zf1_j, zf2_j, ...], le_j)
        else:
            l_recon = self.pl(zf_q_list, leps_q)

        ##### Get parameters for prior
        ## le
        leps_p_loc = torch.zeros(leps_q_loc.size()[0], self.leps_dim).cuda()
        if self.prior_leps_scale_train:
            leps_p_log_scale = F.normalize(torch.sigmoid(self.prior_leps_log_scale), p=2, dim=0)
            leps_p_scale = leps_p_log_scale.reshape(1, -1).repeat(leps_q_loc.size()[0], 1)
        else:
            leps_p_scale = torch.ones(leps_q_loc.size()[0], self.leps_dim).cuda()

        ##### Prior
        pleps = dist.Normal(leps_p_loc, leps_p_scale)  # Normal distribution p(le) - prior

        if self.use_zeps:
            return l_recon, qzf_list, None, zf_q_list, qzeps, None, zeps_q, qleps, pleps, leps_q
                #  l_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q, qleps, pleps, leps_q
        else:
            return l_recon, qzf_list, None, zf_q_list, None, None, None, qleps, pleps, leps_q

    def forward_xl_prior(self, l, x):  # x -> zf, (prior -> le) -> l'
        """
        Generate topographies based on a cell image. Training sceme 2B: sample l from prior (default).
        x -> zf, (prior -> le) -> l' (x -> ze/zf, (prior -> le) -> l')
        """
        ##### Encode; get parameters
        ################################# based on a cell image #############################
        ## ze
        if self.use_zeps:
            zeps_q_loc, zeps_q_scale = self.fdvae.qzeps(x)  # q(ze|x_i) -> qze_loc, qze_scale

        ## zf
        zf_q_locs = []
        zf_q_scales = []
        for k in range(self.fdvae.num_features):
            zfk_q_loc, zfk_q_scale = self.fdvae.qzf_list[k](x)
            zf_q_locs.append(zfk_q_loc)
            zf_q_scales.append(zfk_q_scale)

        ##### Reparameterization trick (encoders)
        ## ze
        if self.use_zeps:
            qzeps = dist.Normal(zeps_q_loc, zeps_q_scale)  # Normal distribution q(ze|x_i)
            zeps_q = qzeps.rsample()  # sample ze_j ~ q(ze|x_i)
        qzf_list = nn.ModuleList()  # Normal distributions [q(zf1|x_i), q(zf2|x_i), ...]

        ## zf
        zf_q_list = []  # samples [zf1_j, zf2_j, ...], where zfk_j ~ q(zfk|x_i)
        for k in range(self.fdvae.num_features):
            qzf_k = NormalModule(zf_q_locs[k], zf_q_scales[k])
            qzf_k.do_rsample()  # sample zfk_j ~ q(zfk|x_i) Normal
            zfk_q = qzf_k.get_rsample()
            qzf_list.append(qzf_k)
            zf_q_list.append(zfk_q)  # only for x_recon
        #####################################################################################

        ##### Get parameters for prior
        ## le
        leps_p_loc = torch.zeros(zfk_q_loc.size()[0], self.leps_dim).cuda()
        if self.prior_leps_scale_train:
            leps_p_log_scale = F.normalize(torch.sigmoid(self.prior_leps_log_scale), p=2, dim=0)
            leps_p_scale = leps_p_log_scale.reshape(1, -1).repeat(zfk_q_loc.size()[0], 1)
        else:
            leps_p_scale = torch.ones(zfk_q_loc.size()[0], self.leps_dim).cuda()

        ##### Prior
        pleps = dist.Normal(leps_p_loc, leps_p_scale)  # Normal distribution p(le) - prior

        ##### Sample le from prior!
        leps_p = pleps.rsample() 

        ##### Decode
        if self.use_zeps:
            l_recon = self.pl(zeps_q, zf_q_list, leps_p)  # l_i_recon ~ p(l|ze_j, [zf1_j, zf2_j, ...], le_j)
        else:
            l_recon = self.pl(zf_q_list, leps_p)


        if self.use_zeps:
            return l_recon, qzf_list, None, zf_q_list, qzeps, None, zeps_q, None, pleps, leps_p
                #  l_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q, qleps, pleps, leps_q
        else:
            return l_recon, qzf_list, None, zf_q_list, None, None, None, None, pleps, leps_p

    def loss_function_x(self, l, x):  # l --TRAIN-> ze/zf/(le) --freeze--> x'
        """
        Step 2 of the training procedure. 
        Here we assume that ALL weights of self.fdvae are frozen
        We train only the encoders qzeps (if zeps is used as part of the topography latent space), qzf_list[k]:

        l ----TRAIN qzf_list ----> zf/(le) --freeze--> x' (l ----TRAIN qzeps qzf_list ----> ze/zf/(le) --freeze--> x')
        """
        x_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q, _, _, _ = self.forward_lx(l, x)  # l -> ze/zf/(le) -> x'

        CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)  # minimize

        ##### NOT USED
        # D = [KL(q(ze|l)q(zf|l) || p(ze)p(zf)) + delta]
        if self.use_zeps:
            D_qzx_pzx = torch.sum(qzeps.log_prob(zeps_q) - pzeps.log_prob(zeps_q))
        else:
            D_qzx_pzx = 0 
        for k in range(self.fdvae.num_features):
            D_qzx_pzx += torch.sum(qzf_list[k].normal.log_prob(zf_q_list[k]) - pzf_list[k].normal.log_prob(zf_q_list[k]))
        D_qzx_pzx += self.delta
    
        if self.kl_llh_multiplier:
            max_objective = -CE_x * D_qzx_pzx  # NOT USED
        else:
        #####
            max_objective = -CE_x

        if self.use_zeps:
            KL_zeps = torch.sum(qzeps.log_prob(zeps_q) - pzeps.log_prob(zeps_q))
            max_objective -= self.lbeta_eps * KL_zeps
        else: 
            KL_zeps = 0

        KLs_zfk = []
        for k in range(self.fdvae.num_features):
            KL_zfk = torch.sum(qzf_list[k].normal.log_prob(zf_q_list[k]) - pzf_list[k].normal.log_prob(zf_q_list[k]))
            max_objective -= self.lbetas_f[k] * KL_zfk
            KLs_zfk.append(KL_zfk)

        loss = -max_objective

        return loss, CE_x, D_qzx_pzx, KL_zeps, KLs_zfk

    def loss_function_l(self, l, x=None):  # x is not needed when we only want to train for the task of topography reconstruction.
        """
        Step 3 of the training procedure.
    
        ALL weights of self.fdvae are still frozen.
        Also we assume that we already trained the encoders qzeps (if zeps is used as part of the topography latent space), 
        qzf_list[k] and FROZE them as well.
        Here we train only the encoder qleps and the decoder pl:
        
        MAIN OBJECTIVE. Reconstruct topography images.
        l ---freeze---------> ze^/zf/---
        l ---TRAIN qleps ---> le -------TRAIN pl---> l'

        REGULARIZE (loss * self.eta). Generate topography images based on a cell image.
        x ---freeze (fdvae) --> ze^/zf/--
                      prior --> le ------TRAIN pl---> l'  

        ^ - if zeps is used as part of the topography latent space (self.use_zeps == True)

        x = (batch_size, 3, 128, 128)
        fs = [(batch_size, 1), (batch_size, N), ...]; fs[k] array corresponds to fk   
        """
        l_recon, qzf_list, pzf_list, zf_q_list, qzeps, pzeps, zeps_q,\
            qleps, pleps, leps_q, qcombined, pcombined, combined_q  = self.forward_l(l)  # l -> ze^/zf/le -> l'

        CE_l = -log_mix_dep_Logistic_256(l, l_recon, average=False, n_comps=10)  # minimize
        max_objective = -CE_l  # ~~ log-likelihood of the data

        if self.use_zeps:
            KL_zeps = torch.sum(qzeps.log_prob(zeps_q) - pzeps.log_prob(zeps_q))  ### FREEZE
            max_objective -= self.lbeta_eps * KL_zeps
        else:
            KL_zeps = 0

        KL_leps = torch.sum(qleps.log_prob(leps_q) - pleps.log_prob(leps_q))  ### TRAIN
        if self.gamma_leps > 0:
            max_objective -= self.gamma_leps * torch.abs(KL_leps - self.c_leps)
        else:
            max_objective -= self.beta_leps * KL_leps

        KLs_zfk = []  ### FREEZE
        for k in range(self.fdvae.num_features):
            KL_zfk = torch.sum(qzf_list[k].normal.log_prob(zf_q_list[k]) - pzf_list[k].normal.log_prob(zf_q_list[k]))
            max_objective -= self.lbetas_f[k] * KL_zfk
            KLs_zfk.append(KL_zfk)

        ##### Combined KL (NOT USED)
        if self.use_zeps:
            comb_q = torch.cat((zeps_q, torch.cat(zf_q_list, dim=1), leps_q), dim=1)
        else:
            comb_q = torch.cat((torch.cat(zf_q_list, dim=1), leps_q), dim=1)

        KL_combined  = torch.sum(qcombined.log_prob(comb_q) - pcombined.log_prob(comb_q))
        max_objective -= self.beta_combined * KL_combined

        KL_combined2 = torch.sum(qcombined.log_prob(combined_q) - pcombined.log_prob(combined_q))
        max_objective -= self.beta_combined2 * KL_combined2
        #####

        CE_l2 = 0
        if x is not None:
            ### training scheme 2Bp (not used)
            # l_recon2, qzf_list2, _, zf_q_list2, qzeps2, _, zeps_q2,\
            #     qleps2, pleps2, leps_q2 = self.forward_xl(l, x)  # x, l -> ze/zf, le -> l'

            ### training scheme 2B (default)
            l_recon2, qzf_list2, _, zf_q_list2, qzeps2, _, zeps_q2,\
                _, pleps2, leps_p = self.forward_xl_prior(l, x)  # x -> zf (prior -> le) -> l'

            CE_l2 = -log_mix_dep_Logistic_256(l, l_recon2, average=False, n_comps=10)  # minimize
            max_objective -= CE_l2 * self.eta  # ~~ log-likelihood of the data, REGULARIZER
        else:
            l_recon2 = None

        loss = -max_objective

        return loss, CE_l, KL_zeps, KL_leps, KLs_zfk, CE_l2, KL_combined, KL_combined2, l_recon, l_recon2


class Discriminator(Encoder):
    def __init__(self):
        super().__init__(1)
        del self.fc11
        del self.fc12
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4,  1)) # probability that the input is real
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.zero_()

    def forward(self, x):

        h = self.conv1(x)
        h = self.bn1(h)
        # activation function is inside of IdResidualConvBlockBN

        h = self.rn1(h)
        h = self.rn2(h)
        h = self.rn3(h)
        h = self.rn4(h)
        h = self.rn5(h)
        h = self.rn6(h)
        h = self.rn7(h)
        h = self.rn8(h)
        h = self.rn9(h)
        h = F.leaky_relu(h)
        h = h.view(-1, 64 * 4 * 4)
        real_prob = nn.Sigmoid()(self.fc(h))

        return real_prob
