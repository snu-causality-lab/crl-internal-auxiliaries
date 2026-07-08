import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch.nn as nn
import numpy as np

def subnet_fc(latent_dim, c_in, c_out, init_identity):
    subnet = nn.Sequential(nn.Linear(c_in, latent_dim), nn.ReLU(),
                            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                            nn.Linear(latent_dim,  c_out))
    if init_identity:
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net(latent_dim, coupling_block, init_identity=True, n_flows=8):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(latent_dim, name='input')]
    for k in range(n_flows):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':lambda c_in,c_out: subnet_fc(latent_dim, c_in, c_out, init_identity), 'clamp':2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                        {'seed':np.random.randint(2**31)},
                        name=F'permute_{k+1}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)


def subnet_conv1(c_in, c_out):
    width = 16
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv2(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def construct_net_image(coupling_block):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(4, 96, 96, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv1, 'clamp':2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv2, 'clamp':2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_fc2, 'clamp':2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)

def subnet_fc2(c_in, c_out):
    width = 392
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet