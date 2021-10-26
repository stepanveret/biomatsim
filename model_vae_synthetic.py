"""
Code is based on: https://github.com/AMLab-Amsterdam/DIVA
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from diva.pixel_cnn_utils import log_mix_dep_Logistic_256
from diva.resnet_blocks_batchnorm import *

# Decoders
class Decoder(nn.Module):  # p(x|z)
    def __init__(self, z_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(z_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        self.rn1 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn2 = nn.Upsample(8)
        self.rn3 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn4 = nn.Upsample(16)
        self.rn5 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn6 = nn.Upsample(32)
        self.rn7 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn8 = nn.Upsample(64)
        self.rn9 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn10 = nn.Upsample(128)
        self.conv1 = nn.Conv2d(64, 100, 3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, 1, padding=0)

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.zero_()

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 64, 4, 4)

        h = self.rn1(h)
        h = self.rn2(h)
        h = self.rn3(h)
        h = self.rn4(h)
        h = self.rn5(h)
        h = self.rn6(h)
        h = self.rn7(h)
        h = self.rn8(h)
        h = self.rn9(h)
        h = self.rn10(h)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        loc_img = self.conv2(h)

        return loc_img


# Encoder
class Encoder(nn.Module):  # q(z|x)
    def __init__(self, z_dim): 
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.rn1 = IdResidualConvBlockBNResize(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvBlockBNIdentity(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvBlockBNResize(32, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn8 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn9 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.fc11 = nn.Sequential(nn.Linear(64 * 4 * 4, z_dim))
        self.fc12 = nn.Sequential(nn.Linear(64 * 4 * 4, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

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
        z_loc = self.fc11(h)
        z_scale = self.fc12(h) + 1e-7

        return z_loc, z_scale


class NormalModule(nn.Module):
    def __init__(self, loc, scale):
        super(NormalModule, self).__init__()
        self.normal = dist.Normal(loc, scale)
        self.rsample = None

    def do_rsample(self):
        self.rsample = self.normal.rsample()

    def get_rsample(self):
        return self.rsample


class VAE(nn.Module):
    def __init__(self, args):

        super(VAE, self).__init__()
        self.z_dim = args.z_dim
        self.decoder = Decoder(self.z_dim)  # p(x|z)
        self.encoder = Encoder(self.z_dim)  # q(z|x)
        self.beta = args.beta  # for KL(q(z|x) || p(z))

        ### https://arxiv.org/abs/1804.03599 approach for z
        self.gamma = args.gamma
        self.c = args.c
        ###

        self.prior_z_scale_train = args.prior_z_scale_train
        self.prior_z_log_scale = nn.Parameter(torch.Tensor([0.0] * int(self.z_dim)))

        self.cuda()

    def forward(self, x):
        """
        x = (batch_size, 3, 128, 128)
        """
        # Encode; get parameters
        z_q_loc, z_q_scale = self.encoder(x)  # q(z|x_i) -> qz_loc, qz_scale

        # Reparameterization trick (encoder)
        qz = dist.Normal(z_q_loc, z_q_scale)  # Normal distribution q(z|x_i)
        z_q = qz.rsample()  # Sample z_j ~ q(z|x_i)

        # Decode
        x_recon = self.decoder(z_q)  # x_i_recon ~ p(x|z_j)

        # Get parameters for prior
        z_p_loc = torch.zeros(z_q_loc.size()[0], self.z_dim).cuda() # prior p(z) -> pz_loc, pz_scale
        if self.prior_z_scale_train:
            z_p_log_scale = F.normalize(torch.sigmoid(self.prior_z_log_scale), p=2, dim=0)
            z_p_scale = z_p_log_scale.reshape(1, -1).repeat(z_q_loc.size()[0], 1)
        else:
            z_p_scale = torch.ones(z_q_loc.size()[0], self.z_dim).cuda()

        # Prior
        pz = dist.Normal(z_p_loc, z_p_scale)  # Normal distribution p(z) - prior

        return x_recon, qz, pz, z_q

    def loss_function(self, x):
        """
        x = (batch_size, 3, 128, 128)
        """
        # supervised
        x_recon, qz, pz, z_q = self.forward(x)

        CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)  # minimize
        max_objective = -CE_x  # ~~ log-likelihood of the data
           
        KL_z = torch.sum(qz.log_prob(z_q) - pz.log_prob(z_q))
            # pz: Normal distribution N(0, I)
            # qz: Normal distribution q(z|x_i); self.encoder(x) -> N(loc(x), scale(x))
            # zeps_q: sample ze_j ~ q(ze|x_i)
            # Given an observation x: KL(q(ze|x)||p(ze)) = E[log q(ze|x) - log p(ze)], ze ~  q(ze|x);
            # Finally, sum by x (batch)

        if self.gamma > 0:
            max_objective -= self.gamma * torch.abs(KL_z - self.c)
        else:
            max_objective -= self.beta * KL_z

        loss = -max_objective

        return loss, CE_x, KL_z, x_recon
