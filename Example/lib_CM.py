import torch
import torch.nn as nn
import numpy as np


class Clustering_Module(nn.Module):

    def __init__(self, input_dim, num_clusters, use_bias=False):
        super(Clustering_Module, self).__init__()
        self.input = input_dim
        self.num_clusters = num_clusters
        self.Id = torch.eye( num_clusters )

        self.gamma = nn.Sequential(
            nn.Linear(input_dim, num_clusters, bias=use_bias),
            nn.Softmax()
        )
        self.mu = nn.Sequential(
            nn.Linear(num_clusters, input_dim, bias=use_bias),
        )

    def _mu(self):
        return self.mu( self.Id )
        
    def predict(self, x):
        return self.gamma(x).argmax(-1)
        
    def predict_proba(self, x):
        return self.gamma(x)
        
    def forward(self, x):
        g  = self.gamma(x)
        tx = self.mu(g)
        u  = self.mu[0].weight.T
        return (x, g, tx, u)


class Clustering_Module_Loss(nn.Module):
    def __init__(self, num_clusters, alpha=1, lbd=1, orth=False, normalize=True, device='cuda'):
        super(Clustering_Module_Loss, self).__init__()
        
        if hasattr( alpha, '__iter__'):
            self.alpha = torch.tensor(alpha, device=device)
            # Use this if alpha are not all the same
            # self.alpha = torch.sort(self.alpha).values
        else:
            self.alpha = torch.ones( num_clusters, device=device ) * alpha
        
        self.lbd   = lbd
        self.orth = orth
        self.normalize = normalize
        self.Id = torch.eye(num_clusters, device=device)
        self.mask = 1-torch.eye(num_clusters, device=device)
        
    def forward(self, inputs, targets=None, split=False):
        x,g,tx,u = inputs
        n,d = x.shape
        k = g.shape[1]
        
        nd = (n*d) if self.normalize else 1.
        
        loss_E1 = torch.sum( torch.square( x - tx ) ) / nd
        
        if self.orth:
            loss_E2 = torch.sum( g*(1-g) ) / nd
            
            uu = torch.matmul( u, u.T )
            loss_E3 = torch.sum( torch.square( uu - self.Id.to(uu.device) ) ) * self.lbd
        else:
            loss_E2 = torch.sum( torch.sum( g*(1-g),0 ) * torch.sum( torch.square(u), 1) ) / nd
            
            gg = torch.matmul( g.T, g)
            uu = torch.matmul( u, u.T )
            gu = gg * uu
            gu = gu * self.mask
            loss_E3 = - torch.sum( gu ) / nd
        
        lmg = torch.log( torch.mean(g,0) +1e-10 )
        # Use this if alpha are not all the same
        # lmg = torch.sort(lmg).values
        loss_E4 = lmg
        
        if split:
            nd  = 1. if self.normalize else n*d
            return torch.stack( (loss_E1/nd, loss_E2/nd, loss_E3/(1 if self.orth else nd) , torch.sum(loss_E4* (1-self.alpha)) ) )
        else:
            return loss_E1 + loss_E2 + loss_E3 + torch.sum( loss_E4 * (1-self.alpha) )


