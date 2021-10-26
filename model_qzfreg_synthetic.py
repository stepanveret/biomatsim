"""
Code is based on: https://github.com/AMLab-Amsterdam/DIVA
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from diva.resnet_blocks_batchnorm import *
from model_vae_synthetic import Encoder


# Conditional prior
class pzf(nn.Module):  # p(zf|f)
    def __init__(self, f_dim, zf_dim): 
        super(pzf, self).__init__()
        # self.fc1 = nn.Sequential(nn.Linear(f_dim, zf_dim, bias=False), nn.BatchNorm1d(zf_dim), nn.LeakyReLU())
        # self.fc21 = nn.Sequential(nn.Linear(zf_dim, zf_dim))
        # self.fc22 = nn.Sequential(nn.Linear(zf_dim, zf_dim), nn.Softplus())
        self.fc1s = nn.BatchNorm1d(f_dim)
        self.fc21s = nn.Sequential(nn.Linear(f_dim, zf_dim))
        self.fc22s = nn.Sequential(nn.Linear(f_dim, zf_dim), nn.Softplus())

        # torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        # self.fc1[1].weight.data.fill_(1)
        # self.fc1[1].bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        # self.fc21[0].bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        # self.fc22[0].bias.data.zero_()

        self.fc1s.weight.data.fill_(1)
        self.fc1s.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21s[0].weight)
        self.fc21s[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22s[0].weight)
        self.fc22s[0].bias.data.zero_()

    def forward(self, f):
        """
        f.shape: (batch_size, 1) or (batch_size, N)
        """
        hidden = self.fc1s(f)
        zf_loc = self.fc21s(hidden)
        zf_scale = self.fc22s(hidden) + 1e-7

        return zf_loc, zf_scale


# Auxiliary tasks
class qf(nn.Module):  # q(f|zf)
    def __init__(self, f_dim, zf_dim, enable_activation=True): 
        super(qf, self).__init__()

        self.fc1 = nn.Linear(zf_dim, f_dim)
        self.activation = nn.LeakyReLU()
        self.enable_activation = enable_activation

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zf):
        if self.enable_activation:
            h = self.activation(zf)
            loc_f = self.fc1(h)
        else:
            loc_f = self.fc1(zf)

        return loc_f


class QZFREG(nn.Module):
    def __init__(self, args):
        """
        zf_dims = [zf1_dim, zf2_dim, ...]
        f_dims = [f1_dim, f2_dim, ...]
        """
        super(QZFREG, self).__init__()

        self.f_dim = args.f_dim
        self.zf_dim = args.zf_dim
        self.qf_act_bool = args.qf_act_bool

        self.pzf = pzf(self.f_dim, self.zf_dim)
        self.qzf = Encoder(self.zf_dim)
        self.qf = qf(self.f_dim, self.zf_dim, self.qf_act_bool)

        self.beta_f = args.beta_f
        self.beta_f_full = args.beta_f_full
        self.aux_loss_multiplier_qzf = args.aux_loss_multiplier_qzf
        self.aux_loss_multiplier_pzf = args.aux_loss_multiplier_pzf

        self.prior_scale_train = args.prior_scale_train
        self.prior_log_scale = nn.Parameter(torch.Tensor([0.0, 0.0]))

        self.cuda()

    def forward(self, x, f):
        """
        x = (batch_size, 3, 128, 128)
        fs = [(batch_size, 1), (batch_size, N), ...]; fs[k] array corresponds to fk
        """
        # Encode; get parameters
        zf_q_loc, zf_q_scale = self.qzf(x)

        # Reparameterization trick (encoder)
        qzf = dist.Normal(zf_q_loc, zf_q_scale)
        zf_q = qzf.rsample() 

        # Get parameters for prior
        zf_p_loc, zf_p_scale = self.pzf(f.float())

        # Reparameterization trick (prior)
        pzf = dist.Normal(zf_p_loc, zf_p_scale)
        zf_p = pzf.rsample()

        zf_p_full_loc = torch.zeros(zf_p_loc.size()[0], self.zf_dim).cuda()

        if self.prior_scale_train:
            zf_p_log_scale = F.normalize(torch.sigmoid(self.prior_log_scale), p=2, dim=0)
            zf_p_full_scale = zf_p_log_scale.reshape(1, -1).repeat(zf_p_loc.size()[0], 1)
        else:
            zf_p_full_scale = torch.ones(zf_p_loc.size()[0], self.zf_dim).cuda()

        # Full prior
        pzf_full = dist.Normal(zf_p_full_loc, zf_p_full_scale)


        # Regression zfq
        f_hat = self.qf(zf_q)

        # Regression zfp
        f_hat2 = self.qf(zf_p)

        return f_hat, f_hat2, qzf, pzf, pzf_full, zf_q, zf_p

    def loss_function(self, x, f):
        """
        x = (batch_size, 3, 128, 128)
        fs = [(batch_size, 1), (batch_size, N), ...]; fs[k] array corresponds to fk
        """
        # supervised
        f_hat, f_hat2, qzf, pzf, pzf_full, zf_q, zf_p = self.forward(x, f)

        max_objective = 0

        MSE = F.mse_loss(f_hat, f.float())
        max_objective -= self.aux_loss_multiplier_qzf * MSE

        MSE2 = F.mse_loss(f_hat2, f.float())
        max_objective -= self.aux_loss_multiplier_pzf * MSE2

        KL_zf = torch.sum(qzf.log_prob(zf_q) - pzf.log_prob(zf_q))
        max_objective -= self.beta_f * KL_zf

        KL_zf_full = torch.sum(pzf.log_prob(zf_p) - pzf_full.log_prob(zf_p))
        max_objective -= self.beta_f_full * KL_zf_full

        loss = -max_objective

        return loss, MSE, MSE2, KL_zf, KL_zf_full
