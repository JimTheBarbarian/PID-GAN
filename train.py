import sys
import argparse
import os
import torch
from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import warnings


from Flow_PID import Generator, K_Generator, Discriminator, Q_Net, Flow_PID


np.random.seed(1234)

# Hyperparameters
num_epochs = 300
lambda_prob = 0.1
lambda_q = 0.5
noise = 0.1


k_hid_dim = 50
k_num_layer = 4

d_hid_dim = 50 
d_num_layer = 3

q_hid_dim = 50
q_num_layer = 4


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

data = np.load('nonlinear2d_data.npz')

X = data['X']
K = data['k']
U = data['u']

N = 10000
N_f = N
N_u = 200
N_b = 100 # for one boundary
q = 1.
u_0 = - 10.
ksat = 10.

X_dim = 1 
Y_dim = 1
Z_dim = 2

L1 = 10.
L2 = 10.


X_u = np.zeros((N_u,2))
Y_u = np.zeros((N_u,1))
X_f = np.zeros((N_f,2))

# Boundary points
x1_b1 = np.zeros(N_b)[:,None]
x2_b1 = L2 * np.random.random(N_b)[:,None]
X_b1 = np.hstack((x1_b1, x2_b1))

x1_b2 = L1 * np.random.random(N_b)[:,None]
x2_b2 = np.zeros(N_b)[:,None]
X_b2 = np.hstack((x1_b2, x2_b2))


x1_b3 = L1 * np.ones(N_b)[:,None]
x2_b3 = L2 * np.random.random(N_b)[:,None]
X_b3 = np.hstack((x1_b3, x2_b3))


x1_b4 = L1 * np.random.random(N_b)[:,None]
x2_b4 = L2 * np.ones(N_b)[:,None]
X_b4 = np.hstack((x1_b4, x2_b4))


X_b = np.hstack((X_b1, X_b2))
X_b = np.hstack((X_b, X_b3))
X_b = np.hstack((X_b, X_b4))


# Collocation points
X1_f = L1 * np.random.random(N_f)[:,None]
X2_f = L2 * np.random.random(N_f)[:,None]
X_f = np.hstack((X1_f, X2_f))

U_data = U
X_data = X

idx_u = np.random.choice(N, N_u, replace=False)
for i in range(N_u):
    X_u[i,:] = X_data[idx_u[i],:]
    Y_u[i,:] = U_data[idx_u[i]]
    
# Normalize data
lb = np.array([0.0, 0.0])
ub = np.array([10.0, 10.0])
lbb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ubb = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
X_u = (X_u - lb) - 0.5*(ub - lb)
X_b = (X_b - lbb) - 0.5*(ubb - lbb)
X_f = (X_f - lb) - 0.5*(ub - lb)


x1_u = X_u[:,0:1]   # dimension  N_u x 1
x2_u = X_u[:,1:2]   # dimension  N_u x 1
y_u = Y_u           # dimension N_u

x1_f = X_f[:,0:1]   # dimension N_f x 1
x2_f = X_f[:,1:2]   # dimension N_f x 1

# Position of the boundary 
x1_b1 = X_b[:,0:1]
x2_b1 = X_b[:,1:2]
x1_b2 = X_b[:,2:3]
x2_b2 = X_b[:,3:4]
x1_b3 = X_b[:,4:5]
x2_b3 = X_b[:,5:6]
x1_b4 = X_b[:,6:7]
x2_b4 = X_b[:,7:8]




# Load Models

D = Discriminator(in_dim = 4, out_dim = 1, 
                  hid_dim = d_hid_dim, 
                  num_layers = d_num_layer
                 ).to(device)


G = Generator()

k_G = K_Generator(in_dim = 1, out_dim = 1, 
              hid_dim = k_hid_dim, 
              num_layers = k_num_layer
             ).to(device)

Q = Q_Net(in_dim = 3, out_dim = 2, 
          hid_dim = q_hid_dim, 
          num_layers = q_num_layer
         ).to(device)




lambdas = [lambda_prob, lambda_q]
pid = Flow_PID(X_u, Y_u, X_b, X_f, N_u, 
                  G, k_G, D, Q, device, num_epochs, 
                  lambdas, noise
                 )


pid.train()


X_star = X
u_star = U.T
k_star = K.T
ksat = 10.

X_star_norm = (X_star - lb) - 0.5*(ub - lb)
u_pred_list = []
f_pred_list = []
k_pred_list = []
for run in range(500):
    u_pred, f_pred, k_pred = pid.predict(X_star_norm)
    k_pred /= ksat
    u_pred_list.append(u_pred)
    f_pred_list.append(f_pred)
    k_pred_list.append(k_pred)

    
u_pred_arr = np.array(u_pred_list)
f_pred_arr = np.array(f_pred_list)
k_pred_arr = np.array(k_pred_list)

uuu_mu_pred = u_pred_arr.mean(axis=0)
fff_mu_pred = f_pred_arr.mean(axis=0)
uuu_Sigma_pred = u_pred_arr.var(axis=0)
fff_Sigma_pred = f_pred_arr.var(axis=0)
kkk_mu_pred = k_pred_arr.mean(axis=0)
kkk_Sigma_pred = k_pred_arr.var(axis=0)

error_u = np.linalg.norm(u_star.T-uuu_mu_pred,2)/np.linalg.norm(u_star.T,2)
error_k = np.linalg.norm(k_star.T-kkk_mu_pred,2)/np.linalg.norm(k_star.T,2)
print('Error u: %e' % (error_u))                     
print('Residual: %e' % (f_pred**2).mean())
print('Error k: %e' % (error_k))


