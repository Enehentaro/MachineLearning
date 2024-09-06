

##############################
       ### Library
##############################

# Standard Library
import os
import sys
import copy
import glob
import time
import random
import pickle
from decimal import Decimal

# External Library
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



##############################
       ### Settings
##############################

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# Set random seed
seed_ = 2434
random.seed(seed_)
torch.manual_seed(seed_)
np.random.seed(seed_)

# Select device
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



##############################
       ### Function
##############################

class PointsDataset(Dataset):
    
    def __init__(self, points, velocity, condition):
        self.points = points
        self.velocity = velocity
        self.condition = condition

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points_item = self.points[idx]
        velocity_item = self.velocity[idx]
        condition_item = self.condition[idx]
        return idx, points_item, velocity_item, condition_item


class BatachDataGenerator():
    
    """
    This is the parent class for inheritance
    This class will be edited in the future.
    """
    
    def __init__(self):
        
        pass


class BatachDataGenerator_Cylinder(BatachDataGenerator):
    
    """
    This is the data generator for cylinder flow
    We call every epoch this instance, and generate residual points for training
    """
    
    def __init__(
        self,
        x_range = [0.0, 24.0],
        y_range = [0.0, 4.0],
        cylinder_center = [2.0, 2.0],
        cylinder_radius = 0.5,
        t_range = [0.0, 10.0],
        u_max = 1.5,
        num_sampling_Xedge = [300, 3000],
        num_sampling_Yedge = [1600, 14000],
        num_sampling_cylinder = [200, 2000],
        num_sampling_region = [10000, 120000],
        chunks = 32,
        initial_coord_CFD=None,
        initial_u_CFD=None
    ):
        
        super(BatachDataGenerator_Cylinder, self).__init__()
        
        self.x_range = x_range
        self.y_range = y_range
        self.cylinder_center = cylinder_center
        self.cylinder_radius = cylinder_radius
        self.t_range = t_range
        self.u_max = u_max
        self.num_sampling_Xedge = num_sampling_Xedge
        self.num_sampling_Yedge = num_sampling_Yedge
        self.num_sampling_cylinder = num_sampling_cylinder
        self.num_sampling_region = num_sampling_region
        self.chunks = chunks
        self.initial_coords_CFD = initial_coord_CFD
        self.initial_u_CFD = initial_u_CFD
        
        self.get_num_sampling()
        
    
    def __call__(self):
        
        if type(self.initial_coords_CFD)==np.ndarray:
            p_init, v_init, c_init = self.initial_generator_withCFD()
        else:
            p_init, v_init, c_init = self.initial_generator()
        p_bound, v_bound, c_bound = self.boundary_generator()
        p_res, v_res, c_res = self.residual_generator()
        
        initial_data = PointsDataset(points=p_init, velocity=v_init, condition=c_init)
        boundary_data = PointsDataset(points=p_bound, velocity=v_bound, condition=c_bound)
        residual_data = PointsDataset(points=p_res, velocity=v_res, condition=c_res)
        
        return initial_data, boundary_data, residual_data
    
    
    def get_num_sampling(self):
        
        self.num_sampling_initial = num_sampling_region[0]
        self.num_sampling_boundary = sum([2*self.num_sampling_Xedge[i]+2*self.num_sampling_Yedge[i]+self.num_sampling_cylinder[i] for i in [0, 1]])
        self.num_sampling_res = num_sampling_region[1]
        
        return
    
    
    def num_sampling(self):
        
        return self.num_sampling_initial, self.num_sampling_boundary, self.num_sampling_res
        
        
    def initial_generator(self):
        
        ### t = t_s
        temp_p = self.sample_from_square_except_circle(self.num_sampling_region[0], self.x_range, self.y_range,  self.cylinder_center, self.cylinder_radius)
        temp_p_ = np.full((self.num_sampling_region[0], 1), self.t_range[0])
        temp_p = np.concatenate([temp_p, temp_p_], axis=1)
        temp_v = np.full((temp_p.shape), 0)
        points_initial = temp_p
        velocity_initial = temp_v
        
        return points_initial, velocity_initial, np.full((len(points_initial), 1), 0)
        
        
    def initial_generator_withCFD(self):
        
        ### t = t_s
        id_array = np.arange(len(self.initial_coords_CFD))
        select_id = np.random.choice(id_array, self.num_sampling_region[0], replace=False)

        temp_p = self.initial_coords_CFD[select_id]
        temp_p_ = np.full((self.num_sampling_region[0], 1), self.t_range[0])
        points_initial  = np.concatenate([temp_p, temp_p_], axis=1)
        velocity_initial = self.initial_u_CFD[select_id]
        
        return points_initial, velocity_initial, np.full((len(points_initial), 1), 0)
    
        
    def boundary_generator(self):
        
        t_min = self.t_range[0]
        t_max = self.t_range[1]
        t_delta = t_max - t_min
        
        points = []
        
        for i in [0, 0, 1]:
        
            temp_p = self.sample_from_line_2d(self.num_sampling_Xedge[i], self.x_range[0], self.y_range)
            temp_v = self.inlet_distribution_u(temp_p[:, 1:2], y_range[1], u_max=1.5)
            temp_c = np.full((len(temp_p), 1), 1)
            points_boundary = temp_p
            velocity_boundary = temp_v
            condition_boundary = temp_c

            temp_p = self.sample_from_line_2d(self.num_sampling_Xedge[i], self.x_range[1], self.y_range)
            temp_v = np.full((temp_p.shape), 0)
            temp_c = np.full((len(temp_p), 1), 2)
            points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
            velocity_boundary = np.concatenate([velocity_boundary, temp_v], axis=0)
            condition_boundary = np.concatenate([condition_boundary, temp_c], axis=0)

            temp_p = self.sample_from_line_2d(self.num_sampling_Yedge[i], self.x_range, self.y_range[0])
            temp_v = np.full((temp_p.shape), 0)
            temp_c = np.full((len(temp_p), 1), 3)
            points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
            velocity_boundary = np.concatenate([velocity_boundary, temp_v], axis=0)
            condition_boundary = np.concatenate([condition_boundary, temp_c], axis=0)

            temp_p = self.sample_from_line_2d(self.num_sampling_Yedge[i], self.x_range, self.y_range[1])
            temp_v = np.full((temp_p.shape), 0)
            temp_c = np.full((len(temp_p), 1), 3)
            points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
            velocity_boundary = np.concatenate([velocity_boundary, temp_v], axis=0)
            condition_boundary = np.concatenate([condition_boundary, temp_c], axis=0)

            temp_p = self.sample_from_circle_edge(self.num_sampling_cylinder[i], self.cylinder_center, self.cylinder_radius)
            temp_v = np.full((temp_p.shape), 0)
            temp_c = np.full((len(temp_p), 1), 3)
            points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
            velocity_boundary = np.concatenate([velocity_boundary, temp_v], axis=0)
            condition_boundary = np.concatenate([condition_boundary, temp_c], axis=0)

            points.append([points_boundary, velocity_boundary, condition_boundary])

        temp_p = np.concatenate([points[0][0], np.full((len(points[0][0]), 1), self.t_range[0])], axis=1)
        points_boundary = temp_p
        velocity_boundary = points[0][1]
        condition_boundary = points[0][2]
        
        temp_p = np.concatenate([points[2][0], (t_delta * np.random.rand(len(points[2][0]), 1) + t_min)], axis=1)
        points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
        velocity_boundary = np.concatenate([velocity_boundary, points[2][1]], axis=0)
        condition_boundary = np.concatenate([condition_boundary, points[2][2]], axis=0)
        
        #temp_p = np.concatenate([points[1][0], np.full((len(points[1][0]), 1), self.t_range[1])], axis=1)
        #points_boundary = np.concatenate([points_boundary, temp_p], axis=0)
        #velocity_boundary = np.concatenate([velocity_boundary, points[1][1]], axis=0)
        #condition_boundary = np.concatenate([condition_boundary, points[1][2]], axis=0)
        
        return points_boundary, velocity_boundary, condition_boundary
    
    
    def residual_generator(self):
        
        t_min = self.t_range[0]
        t_max = self.t_range[1]
        t_delta = t_max - t_min

        """temp_p = self.sample_from_square_except_circle(self.num_sampling_region[1], self.x_range, self.y_range,  self.cylinder_center, self.cylinder_radius)
        temp_p_ = self.sample_from_line_uniform(self.num_sampling_region[1], self.t_range, self.chunks)
        points_residual  = np.concatenate([temp_p, temp_p_], axis=1)"""
        temp_p1 = self.sample_from_square_except_circle(self.num_sampling_region[1]//2, self.x_range, self.y_range, self.cylinder_center, self.cylinder_radius)
        temp_p2 = self.sample_from_square_except_circle(self.num_sampling_region[1]//2, [0,6.0], self.y_range, self.cylinder_center, self.cylinder_radius)
        temp_p = np.concatenate([temp_p1, temp_p2], axis=0)

        rng = np.random.default_rng()
        temp_p = rng.permutation(temp_p, axis=0)

        temp_p_ = self.sample_from_line_uniform(self.num_sampling_region[1], self.t_range, self.chunks)
        points_residual  = np.concatenate([temp_p, temp_p_], axis=1)

        return points_residual , np.full((len(points_residual), 2), 0), np.full((len(points_residual), 1), 0)
    
    
    @classmethod
    def sample_from_square_except_circle(
        cls, 
        num_sampling, 
        x_range,
        y_range, 
        circle_center,
        circle_radius
    ):

        points_array = np.empty(shape=(0,2))

        for i in range(2*num_sampling):

            candidate = np.random.rand(1,2)
            #print(candidate)
            candidate[0][0] = (x_range[1]-x_range[0]) * candidate[0][0] + x_range[0]
            candidate[0][1] = (y_range[1]-y_range[0]) * candidate[0][1] + y_range[0]

            temp = candidate - np.array(circle_center).reshape(1, 2)
            dist2center = np.sqrt(temp[0][0]**2+temp[0][1]**2)

            if dist2center<circle_radius:
                continue
            else:
                points_array = np.concatenate([points_array, candidate], axis=0)


            if len(points_array)==num_sampling:
                break

        if len(points_array)>num_sampling or len(points_array)<num_sampling:
            print("failed to sample points.")
            sys.exit()

        return points_array
    
    
    @classmethod
    def sample_from_circle_edge(cls, num_sampling, circle_center, circle_radius):

        points_array = np.random.rand(num_sampling, 1)

        x_array = circle_radius * np.cos(2*np.pi * points_array)
        y_array = circle_radius * np.sin(2*np.pi * points_array)

        output_array = np.concatenate([x_array.reshape(-1,1), y_array.reshape(-1,1)], axis=1) + circle_center

        return output_array
    
    
    @classmethod
    def sample_from_line(cls, num_sampling, x_range):

        points_array = np.random.rand(num_sampling, 1)

        size_array = np.full((num_sampling, 1), x_range[1]-x_range[0])
        min_array = np.full((num_sampling, 1), x_range[0])

        return points_array * size_array + min_array

                                                         
    @classmethod
    def sample_from_line_uniform(cls, num_sampling, x_range, chunks):
        
        if num_sampling%chunks!=0:
            raise ValueError("num_sampling % chunks != 0.")

        points_array = np.random.rand(num_sampling, 1)

        size_array = np.full((num_sampling, 1), (x_range[1]-x_range[0])/chunks)
        
        dx = (x_range[1]-x_range[0])/chunks
        temp = []
        for i in range(chunks):
            temp = temp + (num_sampling//chunks)*[x_range[0]+dx*i]
        min_array = np.array(temp).reshape(-1, 1)

        return points_array * size_array + min_array
                                                         
    
    @classmethod
    def sample_from_line_2d(cls, num_sampling, x, y):

        if type(x)==list:
            x_point = cls.sample_from_line(num_sampling, x)
            y_point = np.full(x_point.shape, y)

            return np.concatenate([x_point, y_point], axis=1)

        elif type(y)==list:
            y_point = cls.sample_from_line(num_sampling, y)
            x_point = np.full(y_point.shape, x)

            return np.concatenate([x_point, y_point], axis=1)
        
        
    @classmethod
    def inlet_distribution_u(cls, y, y_max, u_max):

        u = (4*u_max*y*(y_max-y))/(y_max**2)
        v = np.full((u.shape), 0)

        return np.concatenate([u, v], axis=1)
        

class ScaledLinear(nn.Linear):
    
    """
    This is the Linear class that corresponds to RWF (Random Weight Factorization).
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(ScaledLinear, self).__init__(in_features, out_features, bias)
        self.scale = nn.Parameter(torch.ones(out_features, 1))

        
    def forward(self, input):
        scaled_weight = self.weight * self.scale
        return nn.functional.linear(input, scaled_weight, self.bias)


class RFFLayer(nn.Module):
    
    """
    This is the sImple RFF (Random Fourier Features) class.
    Mapping is expressed as `f =[cos(2pi*Bx)^T sin(2pi*Bx)^T]^T`.
    """
    
    def __init__(self, input_dim, output_dim):
        super(RFFLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)

        
    def forward(self, x):
        z = x @ self.weight
        
        cos_features = torch.cos(2 * np.pi * z)
        sin_features = torch.sin(2 * np.pi * z)
        global_features = torch.cat([cos_features, sin_features], dim=-1)
        
        return global_features


class ModifiedMLP(nn.Module):
    
    """
    This is the Modified MLP class.
    """
    
    def __init__(self, input_dim, output_dim, num_hidden_layers, num_neurons_per_layer, activation_function, s_mean=1.0, s_std=0.1):
        super(ModifiedMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activation_function = activation_function.lower()
        self.function = self._get_activation_function()

        # Model 
        self.layers.append(RFFLayer(input_dim, num_neurons_per_layer//2))    # Fourier Feature
        
        self.layers.append(ScaledLinear(num_neurons_per_layer, num_neurons_per_layer))    # weight for Residual Connection 1
        self.layers.append(ScaledLinear(num_neurons_per_layer, num_neurons_per_layer))    # weight for Residual Connection 2

        for _ in range(num_hidden_layers-1):
            self.layers.append(ScaledLinear(num_neurons_per_layer, num_neurons_per_layer))

        self.layers.append(nn.Linear(num_neurons_per_layer, output_dim))

        # Initialize weights
        self.kaming_function = ["relu", "gelu"]
        self.xavier_function = ["sigmoid", "tanh"]
        self.s_mean = s_mean
        self.s_std = s_std
        self._initialize_weights()


    def _get_activation_function(self):
        
        if self.activation_function == "relu":
            return torch.nn.functional.relu
        
        elif self.activation_function == "gelu":
            return torch.nn.functional.gelu
        
        elif self.activation_function == "sigmoid":
            return torch.sigmoid
        
        elif self.activation_function == "tanh":
            return torch.tanh
        
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")


    def _initialize_weights(self):
        
        for layer in self.layers[1:]:
            
            if isinstance(layer, nn.Linear):
                
                if self.activation_function in self.kaming_function:
                    nn.init.kaiming_normal_(layer.weight)
                
                elif self.activation_function in self.xavier_function:
                    nn.init.xavier_normal_(layer.weight)
                    
                nn.init.zeros_(layer.bias)
            
            elif isinstance(layer, ScaledLinear):
                
                if self.activation_function in self.kaming_function:
                    nn.init.normal_(layer.scale, mean=self.s_mean, std=self.s_std)
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    with torch.no_grad():
                        layer.weight.data = layer.weight.data / layer.scale

                    nn.init.zeros_(layer.bias)
                
                elif self.activation_function in self.xavier_function:
                    nn.init.normal_(layer.scale, mean=self.s_mean, std=self.s_std)
                    nn.init.xavier_normal_(layer.weight)
                    with torch.no_grad():
                        layer.weight.data = layer.weight.data / layer.scale
                    
                    nn.init.zeros_(layer.bias)


    def forward(self, x):
        
        x = self.layers[0](x)    # spatial fourier feature
        
        u = self.layers[1](x)
        u = self._get_activation_function()(u)    # residual connection 1
        v = self.layers[2](x)
        v = self._get_activation_function()(v)       # residual connection 2
        
        for i, layer in enumerate(self.layers[3:-1]):
            x = layer(x)
            x = self._get_activation_function()(x)
            x = (1-x)*u + x*v

        x = self.layers[-1](x)
        
        return x

        
        
class PhysicsInformedNN_ExpertGuide():
    
    """
    This is Physics-Informed Neural Networks class.
    This class introduces some of the applied techniques introduced in Expert Guide
    """
    
    def __init__(self, device, model, Re, generator, previous_model=None):
        
        ### initialization
        self.device = device
        self.ms_dnn = model.to(device)
        self.Re = Re
        self.generator = generator
        if previous_model is not None:
            self.previous_model = previous_model.to(device)

        ### History
        key = ["initial_u", "initial_v", "initial_p", 
               "boundary_u_in", "boundary_v_in", "boundary_u_out", "boundary_v_out", "boundary_u_wall", "boundary_v_wall", 
               "residual_c", "residual_ns_x", "residual_ns_y", "loss"]
        
        value = [[] for _ in range(len(key))]
        self.loss_history = dict(zip(key, value))

        key = ["ic_u", "ic_v", "ic_p", "bc_u_in", "bc_v_in", "bc_u_out", "bc_v_out", "bc_u_wall", "bc_v_wall", "r_x", "r_y", "r_c"]
        value = [[] for _ in range(len(key))]
        self.loss_weight_history = dict(zip(key, value))

        self.causal_weight_history_rx = []
        self.causal_weight_history_ry = []
        self.causal_weight_history_rc = []
        
        ### epoch information
        self.epoch = 0
        
        ### optimizer
        self.optimizer = torch.optim.Adam(self.ms_dnn.parameters(), lr=0.001)
        #self.optimizer = torch.optim.LBFGS(self.dnn.parameters())
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.90)
        
        ### config
        self.monitor_step = 100
        self.record_step = 100
        self.visualizer_step = 1000
        self.decay = 2000

        self.L = generator.x_range[1] - generator.x_range[0]
        self.W = generator.y_range[1] - generator.y_range[0]
        self.T = generator.t_range[1] - generator.t_range[0]
        
        
    def training_config(self, num_per_chunks, chunks, epsilon, alpha, f):
        
        self.num_per_chunks = num_per_chunks
        self.chunks = chunks
        self.epsilon = epsilon
        self.alpha = alpha
        self.f = f
        
        self.lambda_ic_u = 100.0
        self.lambda_ic_v = 100.0
        self.lambda_ic_p = 100.0

        self.lambda_bc_u_in = 100.0
        self.lambda_bc_v_in = 100.0
        self.lambda_bc_u_out = 1.0
        self.lambda_bc_v_out = 1.0
        self.lambda_bc_u_wall = 10.0
        self.lambda_bc_v_wall = 10.0

        self.lambda_r_x = 1.0
        self.lambda_r_y = 1.0
        self.lambda_r_c = 1.0

        self.limit = 1000

        self.causal_weight_rx = torch.full((num_per_chunks*chunks, 1), 1).to(self.device)
        self.causal_weight_ry = torch.full((num_per_chunks*chunks, 1), 1).to(self.device)
        self.causal_weight_rc = torch.full((num_per_chunks*chunks, 1), 1).to(self.device)
        
        return

        
    def net_u_old(self, x, y, t):  
        
        self.previous_model.eval()

        t = t / self.T  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        
        with torch.no_grad():
            outputs = self.previous_model(torch.cat([x, y, t], dim=1).to(self.device))
            u = outputs[:, 0:1] + 4 * 1.5 * self.W*y * (4.1 - self.W*y) / (4.1**2)
            v = outputs[:, 1:2]
            p = outputs[:, 2:3]
        
        return u, v, p
    

    def net_u(self, x, y, t):  

        t = t / self.T  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        
        outputs = self.ms_dnn(torch.cat([x, y, t], dim=1).to(self.device))
        u = outputs[:, 0:1] + 4 * 1.5 * self.W*y * (4.1 - self.W*y) / (4.1**2)
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        return u, v, p


    def net_ux(self, x, y, t):  
        
        u, v, p = self.net_u(x, y, t)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        
        return u_x, v_x, p_x
    
    
    def net_f(self, x, y, t):

        u, v, p = self.net_u(x, y, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]        
        
        f_x = u_t + u * u_x + v * u_y + p_x - 1/self.Re * (u_xx + u_yy)
        f_y = v_t + u * v_x + v * v_y + p_y - 1/self.Re * (v_xx + v_yy)
        f_continue = u_x + v_y
        
        return f_x, f_y, f_continue
    
    
    def train(self, nIter):
        
        initital_data, boundary_data, residual_data = self.batchdata_generator()
        _, loss_info = self.culc_loss(initital_data, boundary_data, residual_data)

        self.record_history(loss_info)
        
        print("epoch: {:05} | init: {:.8f}, bound: {:.8f}, res: {:.8f}, total: {:.8f}".format(self.epoch, loss_info["initial"], loss_info["boundary"], loss_info["residual"], loss_info["loss"]))
        print("====================================================================================================")
        if self.visualizer_config:
            self.train_visualization()

        for epoch in range(nIter):
            
            self.epoch += 1
            
            self.ms_dnn.train()
            initital_data, boundary_data, residual_data = self.batchdata_generator()
            
            ### update lambda
            if self.epoch%self.f==0:
                self.update_balancing_weight(initital_data, boundary_data, residual_data, scheme="gradnorm")

            ### update causal weight
            if self.epoch%10==0:
                self.update_causal_weight(residual_data)
                
            def closure():
                self.optimizer.zero_grad()

                loss, _ = self.culc_loss(initital_data, boundary_data, residual_data)
                loss.backward()

                return loss

            self.optimizer.step(closure)

            ## calculate the loss again for monitoring
            _, loss_info = self.culc_loss(initital_data, boundary_data, residual_data)
            
            if self.epoch%self.monitor_step==0:
                print("epoch: {:05} | init: {:.8f}, bound: {:.8f}, res: {:.8f}, total: {:.8f}".format(self.epoch, loss_info["initial"], loss_info["boundary"], loss_info["residual"], loss_info["loss"]))
                print("====================================================================================================")
            if self.epoch%self.record_step==0:
                self.record_history(loss_info)

            if self.epoch%self.visualizer_step==0:
                if self.visualizer_config:
                    self.train_visualization()
                    
            if self.epoch%self.decay==0:
                self.scheduler.step()

        return


    def record_history(self, loss_info):

        for key in list(self.loss_history.keys()):
            self.loss_history[key].append(loss_info[key])

        with torch.no_grad():

            self.loss_weight_history["ic_u"].append(self.lambda_ic_u)
            self.loss_weight_history["ic_v"].append(self.lambda_ic_v)
            self.loss_weight_history["ic_p"].append(self.lambda_ic_p)

            self.loss_weight_history["bc_u_in"].append(self.lambda_bc_u_in)
            self.loss_weight_history["bc_v_in"].append(self.lambda_bc_v_in)
            self.loss_weight_history["bc_u_out"].append(self.lambda_bc_u_out)
            self.loss_weight_history["bc_v_out"].append(self.lambda_bc_v_out)
            self.loss_weight_history["bc_u_wall"].append(self.lambda_bc_u_wall)
            self.loss_weight_history["bc_v_wall"].append(self.lambda_bc_v_wall)

            self.loss_weight_history["r_x"].append(self.lambda_r_x)
            self.loss_weight_history["r_y"].append(self.lambda_r_y)
            self.loss_weight_history["r_c"].append(self.lambda_r_c)

            self.causal_weight_history_rx.append(self.causal_weight_rx.cpu().numpy())
            self.causal_weight_history_ry.append(self.causal_weight_ry.cpu().numpy())
            self.causal_weight_history_rc.append(self.causal_weight_rc.cpu().numpy())

        return
    
    
    def batchdata_generator(self):
        
        initital_data, boundary_data, residual_data = self.generator()
        
        if hasattr(self, "previous_model"):
            x_init = torch.tensor(initital_data.points[:, 0:1]).float().to(self.device)
            y_init = torch.tensor(initital_data.points[:, 1:2]).float().to(self.device)
            t_init = torch.full((len(x_init), 1), 1).float().to(self.device)
        
            u_pred_init, v_pred_init, p_pred_init = self.net_u_old(x_init, y_init, t_init)
            with torch.no_grad():
                initital_data.velocity = torch.cat([u_pred_init, v_pred_init, p_pred_init], axis=1).to("cpu").detach().numpy().copy()
                
        return initital_data, boundary_data, residual_data


    def update_balancing_weight(self, initital_data, boundary_data, residual_data, scheme="NTK"):

        if scheme=="NTK":
            self.update_balancing_weight_NTK(initital_data, boundary_data, residual_data)
        elif scheme=="gradnorm":
            self.update_balancing_weight_GradNorm(initital_data, boundary_data, residual_data)

        return

    
    def update_balancing_weight_NTK(self, initital_data, boundary_data, residual_data):

        """
        This function has not updated yet.
        So, don't use this function.
        Update will be soon.        
        """
        
        ## initial loss
        x_init = torch.tensor(initital_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_init = torch.tensor(initital_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_init = torch.tensor(initital_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_init = torch.tensor(initital_data.velocity[:, 0:1]).float().to(self.device)
        v_init = torch.tensor(initital_data.velocity[:, 1:2]).float().to(self.device)
        p_init = torch.tensor(initital_data.velocity[:, 2:3]).float().to(self.device)
            
        _, instance_init_loss_u, instance_init_loss_v, instance_init_loss_p = self.culc_loss_initial(x_init, y_init, t_init, u_init, v_init, p_init, mea=False)

        diagonal_comp_Kic = [[] for _ in range(2)]

        for j, loss in enumerate([instance_init_loss_u, instance_init_loss_v, instance_init_loss_p]):
            for i in range(len(x_init)):

                self.optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

                gradient_vector = []
                for param in self.ms_dnn.parameters():
                    if param.grad is not None:
                        gradient_vector.append(param.grad.view(-1))
                gradient_vector = torch.cat(gradient_vector)

                with torch.no_grad():
                    kic = torch.sum(gradient_vector * gradient_vector)
                    diagonal_comp_Kic[j].append(kic.cpu().numpy().item())

        diagonal_comp_Kic_u = torch.tensor(diagonal_comp_Kic[0])
        diagonal_comp_Kic_v = torch.tensor(diagonal_comp_Kic[1])
        diagonal_comp_Kic_p = torch.tensor(diagonal_comp_Kic[2])
            
        del x_init, y_init, t_init, u_init, v_init
        torch.cuda.empty_cache()
        
        ## boundary loss
        x_bound = torch.tensor(boundary_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_bound = torch.tensor(boundary_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_bound = torch.tensor(boundary_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_bound = torch.tensor(boundary_data.velocity[:, 0:1]).float().to(self.device)
        v_bound = torch.tensor(boundary_data.velocity[:, 1:2]).float().to(self.device)
        cdt = torch.tensor(boundary_data.condition).float().to(self.device)

        _, u_dirichlet_bound_loss, v_dirichlet_bound_loss, u_neumann_bound_loss, v_neumann_bound_loss = self.culc_loss_boundary(x_bound, y_bound, t_bound, u_bound, v_bound, cdt, mea=False)

        diagonal_comp_Kbc = [[] for _ in range(5)]

        for j, loss in enumerate([u_dirichlet_bound_loss, v_dirichlet_bound_loss, u_neumann_bound_loss, v_neumann_bound_loss]):
            for i in range(len(x_bound)):

                self.optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

                gradient_vector = []
                for param in self.ms_dnn.parameters():
                    if param.grad is not None:
                        gradient_vector.append(param.grad.view(-1))
                gradient_vector = torch.cat(gradient_vector)

                with torch.no_grad():
                    kbc = torch.sum(gradient_vector * gradient_vector)
                    diagonal_comp_Kbc[j].append(kbc.cpu().numpy().item())

        diagonal_comp_Kbc_ud = torch.tensor(diagonal_comp_Kbc[0])
        diagonal_comp_Kbc_vd = torch.tensor(diagonal_comp_Kbc[1])
        diagonal_comp_Kbc_un = torch.tensor(diagonal_comp_Kbc[2])
        diagonal_comp_Kbc_vn = torch.tensor(diagonal_comp_Kbc[3])

        del x_bound, y_bound, t_bound, u_bound, v_bound
        torch.cuda.empty_cache()
        
        ## residual loss
        x_res = torch.tensor(residual_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_res = torch.tensor(residual_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_res = torch.tensor(residual_data.points[:, 2:3], requires_grad=True).float().to(self.device)

        _, fx_loss, fy_loss, fc_loss = self.culc_loss_residual(x_res, y_res, t_res, mea=False)
        
        diagonal_comp_Kr = [[] for _ in range(3)]

        for j, loss in enumerate([fx_loss, fy_loss, fc_loss]):
            for i in range(len(x_res)):

                self.optimizer.zero_grad()
                loss[i].backward(retain_graph=True)

                gradient_vector = []
                for param in self.ms_dnn.parameters():
                    if param.grad is not None:
                        gradient_vector.append(param.grad.view(-1))
                gradient_vector = torch.cat(gradient_vector)

                with torch.no_grad():
                    kr = torch.sum(gradient_vector * gradient_vector)
                    diagonal_comp_Kr[j].append(kr.cpu().numpy().item())

        diagonal_comp_Kr_fx = torch.tensor(diagonal_comp_Kr[0])
        diagonal_comp_Kr_fy = torch.tensor(diagonal_comp_Kr[1])
        diagonal_comp_Kr_fc = torch.tensor(diagonal_comp_Kr[2])

        del x_res, y_res, t_res
        torch.cuda.empty_cache()
        
        diagonal_comp_K = torch.cat([diagonal_comp_Kic_u, 
                                     diagonal_comp_Kic_v, 
                                     diagonal_comp_Kic_p,
                                     diagonal_comp_Kbc_ud, 
                                     diagonal_comp_Kbc_vd, 
                                     diagonal_comp_Kbc_un, 
                                     diagonal_comp_Kbc_vn, 
                                     diagonal_comp_Kr_fx, 
                                     diagonal_comp_Kr_fy, 
                                     diagonal_comp_Kr_fc], dim=0) 
        
        new_lambda_ic_u = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kic_u)).cpu().numpy().item()
        new_lambda_ic_v = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kic_v)).cpu().numpy().item()
        new_lambda_ic_p = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kic_p)).cpu().numpy().item()

        new_lambda_bc_ud = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kbc_ud)).cpu().numpy().item()
        new_lambda_bc_vd = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kbc_vd)).cpu().numpy().item()
        new_lambda_bc_un = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kbc_un)).cpu().numpy().item()
        new_lambda_bc_vn = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kbc_vn)).cpu().numpy().item()

        new_lambda_r_u = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kr_fx)).cpu().numpy().item()
        new_lambda_r_v = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kr_fy)).cpu().numpy().item()
        new_lambda_r_p = (torch.sum(diagonal_comp_K)/torch.sum(diagonal_comp_Kr_fc)).cpu().numpy().item()
        
        self.lambda_ic_u = self.alpha * self.lambda_ic_u + (1-self.alpha) * new_lambda_ic_u
        self.lambda_ic_v = self.alpha * self.lambda_ic_v + (1-self.alpha) * new_lambda_ic_v
        self.lambda_ic_p = self.alpha * self.lambda_ic_p + (1-self.alpha) * new_lambda_ic_p

        self.lambda_bc_ud = self.alpha * self.lambda_bc_ud + (1-self.alpha) * new_lambda_bc_ud
        self.lambda_bc_vd = self.alpha * self.lambda_bc_vd + (1-self.alpha) * new_lambda_bc_vd
        self.lambda_bc_un = self.alpha * self.lambda_bc_un + (1-self.alpha) * new_lambda_bc_un
        self.lambda_bc_vn = self.alpha * self.lambda_bc_vn + (1-self.alpha) * new_lambda_bc_vn

        self.lambda_rx = self.alpha * self.lambda_rx + (1-self.alpha) * new_lambda_r_u
        self.lambda_ry = self.alpha * self.lambda_ry + (1-self.alpha) * new_lambda_r_v
        self.lambda_rc = self.alpha * self.lambda_rc + (1-self.alpha) * new_lambda_r_p
        
        return
            

    def update_balancing_weight_GradNorm(self, initital_data, boundary_data, residual_data):

        print("    update balancing weight (epoch: {})".format(self.epoch))

        ## initial loss
        x_init = torch.tensor(initital_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_init = torch.tensor(initital_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_init = torch.tensor(initital_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_init = torch.tensor(initital_data.velocity[:, 0:1]).float().to(self.device)
        v_init = torch.tensor(initital_data.velocity[:, 1:2]).float().to(self.device)
        p_init = torch.tensor(initital_data.velocity[:, 2:3]).float().to(self.device)
            
        _, instance_init_loss_u, instance_init_loss_v, instance_init_loss_p = self.culc_loss_initial(x_init, y_init, t_init, u_init, v_init, p_init)

        grad_ic = []

        for i, loss in enumerate([instance_init_loss_u, instance_init_loss_v, instance_init_loss_p]):
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            gradient_vector = []
            for param in self.ms_dnn.parameters():
                if param.grad is not None:
                    gradient_vector.append(param.grad.view(-1))
            gradient_vector = torch.cat(gradient_vector)

            grad_ic.append(gradient_vector)

        gradnorm_ic_u = torch.norm(grad_ic[0], p=2).cpu().numpy().item()
        gradnorm_ic_v = torch.norm(grad_ic[1], p=2).cpu().numpy().item()
        gradnorm_ic_p = torch.norm(grad_ic[2], p=2).cpu().numpy().item()
            
        del x_init, y_init, t_init, u_init, v_init
        torch.cuda.empty_cache()

        ## boundary loss
        x_bound = torch.tensor(boundary_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_bound = torch.tensor(boundary_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_bound = torch.tensor(boundary_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_bound = torch.tensor(boundary_data.velocity[:, 0:1]).float().to(self.device)
        v_bound = torch.tensor(boundary_data.velocity[:, 1:2]).float().to(self.device)
        cdt = torch.tensor(boundary_data.condition).float().to(self.device)

        _, u_inlet_loss, v_inlet_loss, u_outlet_loss, v_outlet_loss, u_wall_loss, v_wall_loss = self.culc_loss_boundary(x_bound, y_bound, t_bound, u_bound, v_bound, cdt)
        grad_bc = []

        for i, loss in enumerate([u_inlet_loss, v_inlet_loss, u_outlet_loss, v_outlet_loss, u_wall_loss, v_wall_loss]):
                
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            gradient_vector = []
            for param in self.ms_dnn.parameters():
                if param.grad is not None:
                    gradient_vector.append(param.grad.view(-1))
            gradient_vector = torch.cat(gradient_vector)

            grad_bc.append(gradient_vector)

        gradnorm_bc_u_in = torch.norm(grad_bc[0], p=2).cpu().numpy().item()
        gradnorm_bc_v_in = torch.norm(grad_bc[1], p=2).cpu().numpy().item()
        gradnorm_bc_u_out = torch.norm(grad_bc[2], p=2).cpu().numpy().item()
        gradnorm_bc_v_out = torch.norm(grad_bc[3], p=2).cpu().numpy().item()
        gradnorm_bc_u_wall = torch.norm(grad_bc[4], p=2).cpu().numpy().item()
        gradnorm_bc_v_wall = torch.norm(grad_bc[5], p=2).cpu().numpy().item()

        del x_bound, y_bound, t_bound, u_bound, v_bound
        torch.cuda.empty_cache()

        ## residual loss
        x_res = torch.tensor(residual_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_res = torch.tensor(residual_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_res = torch.tensor(residual_data.points[:, 2:3], requires_grad=True).float().to(self.device)

        _, fx_loss, fy_loss, fc_loss = self.culc_loss_residual(x_res, y_res, t_res)

        grad_res = []

        for i, loss in enumerate([fx_loss, fy_loss, fc_loss]):
                    
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            gradient_vector = []
            for param in self.ms_dnn.parameters():
                if param.grad is not None:
                    gradient_vector.append(param.grad.view(-1))
            gradient_vector = torch.cat(gradient_vector)

            grad_res.append(gradient_vector)

        gradnorm_res_fx = torch.norm(grad_res[0], p=2).cpu().numpy().item()
        gradnorm_res_fy = torch.norm(grad_res[1], p=2).cpu().numpy().item()
        gradnorm_res_fc = torch.norm(grad_res[2], p=2).cpu().numpy().item()

        del x_res, y_res, t_res
        torch.cuda.empty_cache()

        gradnorm = (gradnorm_ic_u + gradnorm_ic_v + gradnorm_ic_p 
                    + gradnorm_bc_u_in + gradnorm_bc_v_in + gradnorm_bc_u_out + gradnorm_bc_v_out + gradnorm_bc_u_wall + gradnorm_bc_v_wall 
                    + gradnorm_res_fx + gradnorm_res_fy + gradnorm_res_fc)/12
        #gradnorm = torch.mean(torch.cat([gradnorm_ic_u, gradnorm_ic_v, gradnorm_ic_p, gradnorm_bc_ud, gradnorm_bc_vd, gradnorm_bc_un, gradnorm_bc_vn, gradnorm_res_fx, gradnorm_res_fy, gradnorm_res_fc], dim=0))

        print("     gradnorm_ic_u: {:.8f}".format(gradnorm_ic_u))
        print("     gradnorm_ic_v: {:.8f}".format(gradnorm_ic_v))
        print("     gradnorm_ic_p: {:.8f}".format(gradnorm_ic_p))

        print("     gradnorm_bc_u_in: {:.8f}".format(gradnorm_bc_u_in))
        print("     gradnorm_bc_v_in: {:.8f}".format(gradnorm_bc_v_in))
        print("     gradnorm_bc_u_out: {:.8f}".format(gradnorm_bc_u_out))
        print("     gradnorm_bc_v_out: {:.8f}".format(gradnorm_bc_v_out))
        print("     gradnorm_bc_u_wall: {:.8f}".format(gradnorm_bc_u_wall))
        print("     gradnorm_bc_v_wall: {:.8f}".format(gradnorm_bc_v_wall))

        print("     gradnorm_res_fx: {:.8f}".format(gradnorm_res_fx))
        print("     gradnorm_res_fy: {:.8f}".format(gradnorm_res_fy))
        print("     gradnorm_res_fc: {:.8f}".format(gradnorm_res_fc))

        print("     gradnorm: {:.8f}".format(gradnorm))

        new_lambda_ic_u = gradnorm / gradnorm_ic_u
        new_lambda_ic_v = gradnorm / gradnorm_ic_v
        new_lambda_ic_p = gradnorm / gradnorm_ic_p

        new_lambda_bc_u_in = gradnorm / gradnorm_bc_u_in
        new_lambda_bc_v_in = gradnorm / gradnorm_bc_v_in
        new_lambda_bc_u_out = gradnorm / gradnorm_bc_u_out
        new_lambda_bc_v_out = gradnorm / gradnorm_bc_v_out
        new_lambda_bc_u_wall = gradnorm / gradnorm_bc_u_wall
        new_lambda_bc_v_wall = gradnorm / gradnorm_bc_v_wall

        new_lambda_r_x = gradnorm / gradnorm_res_fx
        new_lambda_r_y = gradnorm / gradnorm_res_fy
        new_lambda_r_c = gradnorm / gradnorm_res_fc

        print("     lambda_ic_u: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_ic_u, new_lambda_ic_u, min(self.alpha * self.lambda_ic_u + (1-self.alpha) * new_lambda_ic_u ,self.limit)))
        print("     lambda_ic_v: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_ic_v, new_lambda_ic_v, min(self.alpha * self.lambda_ic_v + (1-self.alpha) * new_lambda_ic_v ,self.limit)))
        print("     lambda_ic_p: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_ic_p, new_lambda_ic_p, min(self.alpha * self.lambda_ic_p + (1-self.alpha) * new_lambda_ic_p ,self.limit)))

        print("     lambda_bc_u_in: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_u_in, new_lambda_bc_u_in, min(self.alpha * self.lambda_bc_u_in + (1-self.alpha) * new_lambda_bc_u_in ,self.limit)))
        print("     lambda_bc_v_in: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_v_in, new_lambda_bc_v_in, min(self.alpha * self.lambda_bc_v_in + (1-self.alpha) * new_lambda_bc_v_in ,self.limit)))
        print("     lambda_bc_u_out: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_u_out, new_lambda_bc_u_out, min(self.alpha * self.lambda_bc_u_out + (1-self.alpha) * new_lambda_bc_u_out ,self.limit)))
        print("     lambda_bc_v_out: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_v_out, new_lambda_bc_v_out, min(self.alpha * self.lambda_bc_v_out + (1-self.alpha) * new_lambda_bc_v_out ,self.limit)))
        print("     lambda_bc_u_wall: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_u_wall, new_lambda_bc_u_wall, min(self.alpha * self.lambda_bc_u_wall + (1-self.alpha) * new_lambda_bc_u_wall ,self.limit)))
        print("     lambda_bc_v_wall: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_bc_v_wall, new_lambda_bc_v_wall, min(self.alpha * self.lambda_bc_v_wall + (1-self.alpha) * new_lambda_bc_v_wall ,self.limit)))

        print("     lambda_rx: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_r_x, new_lambda_r_x, min(self.alpha * self.lambda_r_x + (1-self.alpha) * new_lambda_r_x ,self.limit)))
        print("     lambda_ry: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_r_y, new_lambda_r_y, min(self.alpha * self.lambda_r_y + (1-self.alpha) * new_lambda_r_y ,self.limit)))
        print("     lambda_rc: {:.8f} ({:.8f}) --> {:.8f}".format(self.lambda_r_c, new_lambda_r_c, min(self.alpha * self.lambda_r_c + (1-self.alpha) * new_lambda_r_c ,self.limit)))

        self.lambda_ic_u = min(self.alpha * self.lambda_ic_u + (1-self.alpha) * new_lambda_ic_u, self.limit)
        self.lambda_ic_v = min(self.alpha * self.lambda_ic_v + (1-self.alpha) * new_lambda_ic_v, self.limit)
        self.lambda_ic_p = min(self.alpha * self.lambda_ic_p + (1-self.alpha) * new_lambda_ic_p, self.limit)

        self.lambda_bc_u_in = min(self.alpha * self.lambda_bc_u_in + (1-self.alpha) * new_lambda_bc_u_in, self.limit)
        self.lambda_bc_v_in = min(self.alpha * self.lambda_bc_v_in + (1-self.alpha) * new_lambda_bc_v_in, self.limit)
        self.lambda_bc_u_out = min(self.alpha * self.lambda_bc_u_out + (1-self.alpha) * new_lambda_bc_u_out, self.limit)
        self.lambda_bc_v_out = min(self.alpha * self.lambda_bc_v_out + (1-self.alpha) * new_lambda_bc_v_out, self.limit)
        self.lambda_bc_u_wall = min(self.alpha * self.lambda_bc_u_wall + (1-self.alpha) * new_lambda_bc_u_wall, self.limit)
        self.lambda_bc_v_wall = min(self.alpha * self.lambda_bc_v_wall + (1-self.alpha) * new_lambda_bc_v_wall, self.limit)

        self.lambda_rx = min(self.alpha * self.lambda_r_x + (1-self.alpha) * new_lambda_r_x, self.limit)
        self.lambda_ry = min(self.alpha * self.lambda_r_y + (1-self.alpha) * new_lambda_r_y, self.limit)
        self.lambda_rc = min(self.alpha * self.lambda_r_c + (1-self.alpha) * new_lambda_r_c, self.limit)
        
        print("====================================================================================================")

        return

    
    def update_causal_weight(self, residual_data):

        if self.epoch%10==0:
            print("    update causal weight (epoch: {})".format(self.epoch))

        x_res = torch.tensor(residual_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_res = torch.tensor(residual_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_res = torch.tensor(residual_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        
        fx_loss = []
        fy_loss = []
        fc_loss = []

        for i in range(0, self.chunks):

            _, fx_loss_, fy_loss_, fc_loss_ = self.culc_loss_residual(x_res[i*self.num_per_chunks:(i+1)*self.num_per_chunks], y_res[i*self.num_per_chunks:(i+1)*self.num_per_chunks], t_res[i*self.num_per_chunks:(i+1)*self.num_per_chunks], mea=True, causality=False)
            fx_loss.append(fx_loss_)
            fy_loss.append(fy_loss_)
            fc_loss.append(fc_loss_)

        fx_loss = torch.tensor(fx_loss)
        fy_loss = torch.tensor(fy_loss)
        fc_loss = torch.tensor(fc_loss)
        
        with torch.no_grad():

            new_weight_map = {}

            for key,loss in zip(["rx", "ry", "rc"], [fx_loss, fy_loss, fc_loss]):

                if self.epoch%10==0:
                    print("     target: {}".format(key))

                new_weight = []
                
                for i in range(0, self.chunks):
                    
                    if self.epoch%10==0:
                        print("     chunk-{}: {}, {}".format(i, torch.sum(loss[0:i]), torch.exp(-self.epsilon*torch.sum(loss[0:i]))))
                    
                    for j in range(self.num_per_chunks):
                    
                        #id = i*self.num_per_chunks + j
                        #self.causal_weight[id, 0] = torch.exp(-self.epsilon*torch.sum(res_loss[0:i*self.num_per_chunks])).cpu().numpy().item()
                        if i==0:
                            new_weight.append(1)
                        else:
                            new_weight.append(torch.exp(-self.epsilon*torch.sum(loss[0:i])).cpu().numpy().item())
                        #print(id, self.causal_weight[id, 0], torch.exp(-self.epsilon*torch.sum(res_loss[0:i*self.num_per_chunks])))
                            
                new_weight_map[key] = new_weight

        self.causal_weight_rx = torch.tensor(new_weight_map["rx"]).view(-1, 1).to(self.device)
        self.causal_weight_ry = torch.tensor(new_weight_map["ry"]).view(-1, 1).to(self.device)
        self.causal_weight_rc = torch.tensor(new_weight_map["rc"]).view(-1, 1).to(self.device)

        if self.epoch%10==0:
            print("====================================================================================================")
        
        return
    
    
    def culc_loss(self, initital_data, boundary_data, residual_data):
        
        """
        ===== boundary condition =====
          1. inlet (velocity dirichlet)
          2. outlet (velocity neumann)
          3. wall (velocity dirichlet)
        
        """

        loss_info = {}
        
        ## initial loss
        x_init = torch.tensor(initital_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_init = torch.tensor(initital_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_init = torch.tensor(initital_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_init = torch.tensor(initital_data.velocity[:, 0:1]).float().to(self.device)
        v_init = torch.tensor(initital_data.velocity[:, 1:2]).float().to(self.device)
        p_init = torch.tensor(initital_data.velocity[:, 2:3]).float().to(self.device)
        
        _, u_init_loss, v_init_loss, p_init_loss = self.culc_loss_initial(x_init, y_init, t_init, u_init, v_init, p_init)
        init_loss = self.lambda_ic_u * u_init_loss + self.lambda_ic_v * v_init_loss + self.lambda_ic_p * p_init_loss

        with torch.no_grad():
            loss_info["initial_u"] = u_init_loss.cpu().numpy().item(),
            loss_info["initial_v"] = v_init_loss.cpu().numpy().item(),
            loss_info["initial_p"] = p_init_loss.cpu().numpy().item(),
            loss_info["initial"] = init_loss.cpu().numpy().item()
            
        del x_init, y_init, t_init, u_init, v_init
        torch.cuda.empty_cache()
        
        ## boundary loss
        x_bound = torch.tensor(boundary_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_bound = torch.tensor(boundary_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_bound = torch.tensor(boundary_data.points[:, 2:3], requires_grad=True).float().to(self.device)
        u_bound = torch.tensor(boundary_data.velocity[:, 0:1]).float().to(self.device)
        v_bound = torch.tensor(boundary_data.velocity[:, 1:2]).float().to(self.device)
        cdt = torch.tensor(boundary_data.condition).float().to(self.device)

        _, u_inlet_loss, v_inlet_loss, u_outlet_loss, v_outlet_loss, u_wall_loss, v_wall_loss = self.culc_loss_boundary(x_bound, y_bound, t_bound, u_bound, v_bound, cdt)
        bound_loss = self.lambda_bc_u_in * u_inlet_loss + self.lambda_bc_v_in * v_inlet_loss + self.lambda_bc_u_out * u_outlet_loss + self.lambda_bc_v_out * v_outlet_loss + self.lambda_bc_u_wall * u_wall_loss + self.lambda_bc_v_wall * v_wall_loss

        with torch.no_grad():
            loss_info["boundary_u_in"] = u_inlet_loss.cpu().numpy().item()
            loss_info["boundary_v_in"] = v_inlet_loss.cpu().numpy().item()
            loss_info["boundary_u_out"] = u_outlet_loss.cpu().numpy().item()
            loss_info["boundary_v_out"] = v_outlet_loss.cpu().numpy().item()
            loss_info["boundary_u_wall"] = u_wall_loss.cpu().numpy().item()
            loss_info["boundary_v_wall"] = v_wall_loss.cpu().numpy().item()
            loss_info["boundary"] = bound_loss.cpu().numpy().item()

        del x_bound, y_bound, t_bound, u_bound, v_bound
        torch.cuda.empty_cache()
        
        ## residual loss
        x_res = torch.tensor(residual_data.points[:, 0:1], requires_grad=True).float().to(self.device)
        y_res = torch.tensor(residual_data.points[:, 1:2], requires_grad=True).float().to(self.device)
        t_res = torch.tensor(residual_data.points[:, 2:3], requires_grad=True).float().to(self.device)

        _, fx_loss, fy_loss, fc_loss = self.culc_loss_residual(x_res, y_res, t_res)
        res_loss = self.lambda_r_x * fx_loss + self.lambda_r_y * fy_loss + self.lambda_r_c * fc_loss
        
        with torch.no_grad():
            loss_info["residual_ns_x"] = fx_loss.cpu().numpy().item()
            loss_info["residual_ns_y"] = fy_loss.cpu().numpy().item()
            loss_info["residual_c"] = fc_loss.cpu().numpy().item()
            loss_info["residual"] = res_loss.cpu().numpy().item()

        loss = init_loss + bound_loss + res_loss

        with torch.no_grad():
            loss_info["loss"] = loss.cpu().numpy().item()
        
        return loss, loss_info
    
    
    def culc_loss_initial(self, x_init, y_init, t_init, u_init, v_init, p_init, mea=True):
        
        if mea==True:
            u_pred_init, v_pred_init, p_pred_init = self.net_u(x_init, y_init, t_init)
            u_init_loss = torch.mean((u_init - u_pred_init) ** 2)
            v_init_loss = torch.mean((v_init - v_pred_init) ** 2)
            p_init_loss = torch.mean((p_init - p_pred_init) ** 2)
        
            init_loss = u_init_loss + v_init_loss + p_init_loss

        else:
            u_pred_init, v_pred_init, p_pred_init = self.net_u(x_init, y_init, t_init)
            u_init_loss = (u_init - u_pred_init) ** 2
            v_init_loss = (v_init - v_pred_init) ** 2
            p_init_loss = (p_init - p_pred_init) ** 2
        
            init_loss = u_init_loss + v_init_loss
        
        return init_loss, u_init_loss, v_init_loss, p_init_loss
    
    
    def culc_loss_boundary(self, x_bound, y_bound, t_bound, u_bound, v_bound, cdt, mea=True):

        if mea==True:
            ## inlet boundary
            x_bound_inlet = x_bound[cdt[:,0]==1]
            y_bound_inlet = y_bound[cdt[:,0]==1]
            t_bound_inlet = t_bound[cdt[:,0]==1]
            u_bound_inlet = u_bound[cdt[:,0]==1]
            v_bound_inlet = v_bound[cdt[:,0]==1]

            u_pred_bound_inlet, v_pred_bound_inlet, p_pred_bound_inlet = self.net_u(x_bound_inlet, y_bound_inlet, t_bound_inlet)

            u_inlet_loss = torch.mean((u_bound_inlet - u_pred_bound_inlet) ** 2)
            v_inlet_loss = torch.mean((v_bound_inlet - v_pred_bound_inlet) ** 2)

            ## outlet boundary
            x_bound_outlet = x_bound[cdt[:,0]==2]
            y_bound_outlet = y_bound[cdt[:,0]==2]
            t_bound_outlet = t_bound[cdt[:,0]==2]

            _, _, p_pred_bound_outlet = self.net_u(x_bound_outlet, y_bound_outlet, t_bound_outlet)
            u_pred_bound_outlet, v_pred_bound_outlet, _ = self.net_ux(x_bound_outlet, y_bound_outlet, t_bound_outlet)

            u_outlet_loss = torch.mean(((1/self.Re)*u_pred_bound_outlet-p_pred_bound_outlet) ** 2)
            v_outlet_loss = torch.mean((v_pred_bound_outlet) ** 2)

            ## wall boundary
            x_bound_wall = x_bound[cdt[:,0]==3]
            y_bound_wall = y_bound[cdt[:,0]==3]
            t_bound_wall = t_bound[cdt[:,0]==3]

            u_pred_bound_wall, v_pred_bound_wall, _ = self.net_u(x_bound_wall, y_bound_wall, t_bound_wall)

            u_wall_loss = torch.mean((u_pred_bound_wall) ** 2)
            v_wall_loss = torch.mean((v_pred_bound_wall) ** 2)

        else:
            ## inlet boundary
            x_bound_inlet = x_bound[cdt[:,0]==1]
            y_bound_inlet = y_bound[cdt[:,0]==1]
            t_bound_inlet = t_bound[cdt[:,0]==1]
            u_bound_inlet = u_bound[cdt[:,0]==1]
            v_bound_inlet = v_bound[cdt[:,0]==1]

            u_pred_bound_inlet, v_pred_bound_inlet, p_pred_bound_inlet = self.net_u(x_bound_inlet, y_bound_inlet, t_bound_inlet)

            u_inlet_loss = (u_bound_inlet - u_pred_bound_inlet) ** 2
            v_inlet_loss = (v_bound_inlet - v_pred_bound_inlet) ** 2

            ## outlet boundary
            x_bound_outlet = x_bound[cdt[:,0]==2]
            y_bound_outlet = y_bound[cdt[:,0]==2]
            t_bound_outlet = t_bound[cdt[:,0]==2]

            _, _, p_pred_bound_outlet = self.net_u(x_bound_outlet, y_bound_outlet, t_bound_outlet)
            u_pred_bound_outlet, v_pred_bound_outlet, _ = self.net_ux(x_bound_outlet, y_bound_outlet, t_bound_outlet)

            u_outlet_loss = ((1/self.Re)*u_pred_bound_outlet-p_pred_bound_outlet) ** 2
            v_outlet_loss = (v_pred_bound_outlet) ** 2

            ## wall boundary
            x_bound_wall = x_bound[cdt[:,0]==3]
            y_bound_wall = y_bound[cdt[:,0]==3]
            t_bound_wall = t_bound[cdt[:,0]==3]

            u_pred_bound_wall, v_pred_bound_wall, _ = self.net_u(x_bound_wall, y_bound_wall, t_bound_wall)

            u_wall_loss = (u_pred_bound_wall) ** 2
            v_wall_loss = (v_pred_bound_wall) ** 2

        bound_loss = u_inlet_loss + v_inlet_loss + u_outlet_loss + v_outlet_loss + u_wall_loss + v_wall_loss
        
        return bound_loss, u_inlet_loss, v_inlet_loss, u_outlet_loss, v_outlet_loss, u_wall_loss, v_wall_loss
    
    
    def culc_loss_residual(self, x_res, y_res, t_res, mea=True, causality=True):

        if mea==True:

            f_pred_x, f_pred_y, f_pred_continue = self.net_f(x_res, y_res, t_res) 

            if causality==True:
                fx_loss = torch.mean(self.causal_weight_rx * f_pred_x ** 2)
                fy_loss = torch.mean(self.causal_weight_ry * f_pred_y ** 2)
                fc_loss = torch.mean(self.causal_weight_rc * f_pred_continue ** 2)

            else:
                fx_loss = torch.mean(f_pred_x ** 2)
                fy_loss = torch.mean(f_pred_y ** 2)
                fc_loss = torch.mean(f_pred_continue ** 2)

            res_loss = fx_loss + fy_loss + fc_loss
        
        else:

            f_pred_x, f_pred_y, f_pred_continue = self.net_f(x_res, y_res, t_res) 
            
            if causality==True:
                fx_loss = self.causal_weight_rx * f_pred_x ** 2
                fy_loss = self.causal_weight_ry * f_pred_y ** 2
                fc_loss = self.causal_weight_rc * f_pred_continue ** 2
            else:
                fx_loss = f_pred_x ** 2
                fy_loss = f_pred_y ** 2
                fc_loss = f_pred_continue ** 2

            res_loss = fx_loss + fy_loss + fc_loss
        
        return res_loss, fx_loss, fy_loss, fc_loss

    
    def set_train_visualizer(self, output_dir, x_range, y_range, t_range, resolution, cylinder_center, cylinder_radius, min_max, label):

        self.visualizer_config = {
            "output_dir": output_dir,
            "x_range": x_range,
            "y_range": y_range,
            "t_range": t_range,
            "resolution": resolution,
            "cylinder_center": cylinder_center,
            "cylinder_radius": cylinder_radius,
            "min_max": min_max,
            "label": label
        }
                
        
    def train_visualization(self):

        output_dir = self.visualizer_config["output_dir"]
        x_range = self.visualizer_config["x_range"]
        y_range = self.visualizer_config["y_range"]
        t_range = self.visualizer_config["t_range"]
        resolution = self.visualizer_config["resolution"]
        cylinder_center = self.visualizer_config["cylinder_center"]
        cylinder_radius = self.visualizer_config["cylinder_radius"]
        min_max = self.visualizer_config["min_max"]
        label = self.visualizer_config["label"]

        path2loss = os.path.join(output_dir, "loss")
        path2weight = os.path.join(output_dir, "weight")
        path2predict = os.path.join(output_dir, "predict")

        if not os.path.exists(path2loss):
            os.makedirs(path2loss)
        if not os.path.exists(path2weight):
            os.makedirs(path2weight)
        if not os.path.exists(path2predict):
            os.makedirs(path2predict)

        ### loss curve
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        
        ax.set_yscale("log")
        ax.plot(self.loss_history["initial_u"], label="ic_u")
        ax.plot(self.loss_history["initial_v"], label="ic_v")
        ax.plot(self.loss_history["initial_p"], label="ic_p")
        ax.plot(self.loss_history["boundary_u_in"], label="bc_u_in")
        ax.plot(self.loss_history["boundary_v_in"], label="bc_v_in")
        ax.plot(self.loss_history["boundary_u_out"], label="bc_u_out")
        ax.plot(self.loss_history["boundary_v_out"], label="bc_v_out")
        ax.plot(self.loss_history["boundary_u_wall"], label="bc_u_wall")
        ax.plot(self.loss_history["boundary_v_wall"], label="bc_v_wall")
        ax.plot(self.loss_history["residual_ns_x"], label="r_x")
        ax.plot(self.loss_history["residual_ns_y"], label="r_y")
        ax.plot(self.loss_history["residual_c"], label="r_c")
        ax.plot(self.loss_history["loss"], linewidth=3, label="total")
        ax.legend(loc="best")
        fig.savefig(os.path.join(path2loss, "loss_{:05}.png".format(self.epoch)), dpi=100)
        
        plt.close(fig)

        ### current causal weight
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(self.causal_weight_history_rx[-1], label="r_x")
        ax.plot(self.causal_weight_history_ry[-1], label="r_y")
        ax.plot(self.causal_weight_history_rc[-1], label="r_c")
        ax.set_ylim([-0.05, 1.05])
        fig.savefig(os.path.join(path2weight, "current_causal_weight_{:05}.png".format(self.epoch)), dpi=100)
        
        plt.close(fig)

        ### causal weight
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot([x.min() for x in self.causal_weight_history_rx], label="r_x")
        ax.plot([x.min() for x in self.causal_weight_history_ry], label="r_y")
        ax.plot([x.min() for x in self.causal_weight_history_rc], label="r_c")
        ax.set_ylim([-0.05, 1.05])
        fig.savefig(os.path.join(path2weight, "causal_weight_{:05}.png".format(self.epoch)), dpi=100)
        
        plt.close(fig)

        ### loss weight
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(self.loss_weight_history["ic_u"], label="ic_u")
        ax.plot(self.loss_weight_history["ic_v"], label="ic_v")
        ax.plot(self.loss_weight_history["ic_p"], label="ic_p")
        ax.plot(self.loss_weight_history["bc_u_in"], label="bc_u_in")
        ax.plot(self.loss_weight_history["bc_v_in"], label="bc_v_in")
        ax.plot(self.loss_weight_history["bc_u_out"], label="bc_u_out")
        ax.plot(self.loss_weight_history["bc_v_out"], label="bc_v_out")
        ax.plot(self.loss_weight_history["bc_u_wall"], label="bc_u_wall")

        ax.plot(self.loss_weight_history["r_x"], label="rx")
        ax.plot(self.loss_weight_history["r_y"], label="ry")
        ax.plot(self.loss_weight_history["r_c"], label="rc")
        ax.legend(loc="best")
        fig.savefig(os.path.join(path2weight, "loss_weight_{:05}.png".format(self.epoch)), dpi=100)
        
        plt.close(fig)

        ### prediction
        x = np.linspace(x_range[0], x_range[1], resolution[0])
        y = np.linspace(y_range[0], y_range[1], resolution[1])
        #x = np.linspace(0, 1, resolution[0])
        #y = np.linspace(0, 1, resolution[1])
        xx, yy = np.meshgrid(x, y)
        
        #t_list = [[t_range[0], "s"], [(t_range[1]-t_range[0])/2, "m"], [t_range[1], "f"]]
        t_list = [[0, "s"], [0.5, "m"], [1, "f"]]

        for t in t_list:

            temp = np.concatenate([xx.reshape((-1,1)),yy.reshape((-1,1))], axis=1)
            temp_ = np.concatenate([temp, np.full((len(temp),1), t[0])], axis=1)
            u, v, p = self.predict(temp_)
            #temp[:,0:1] = temp[:,0:1] * self.L
            #temp[:,1:2] = temp[:,1:2] * self.W

            if label=="velocity_x":
                mmm = u
            elif label=="velocity_y":
                mmm = v
            elif label=="pressure":
                mmm = p
            elif label=="velocity_mag":
                mmm = np.sqrt(u**2 + v**2)

            for i in range(len(mmm)):
                if  np.linalg.norm(temp[i]-np.array(cylinder_center)) < cylinder_radius:
                    mmm[i] = np.nan
            mmm = np.flipud(mmm.reshape(resolution[1], resolution[0]))

            figure = plot_Scalar2D(xx, yy, mmm, vmin=min_max[0], vmax=min_max[1], label=label)
            figure.savefig(os.path.join(path2predict, "output_{}_{:05}.png".format(t[1], self.epoch)), dpi=400)

            plt.close(figure)
        
            
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        
        self.ms_dnn.eval()
        u, v, p = self.net_u(x, y, t)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        
        return u, v, p

def plot_Scalar2D(
    X, Y, img, split_x=5, split_y=4, vmin=-1, vmax=1, label="velocity"
):
    """
    docstring
    """
    
    fig = plt.figure(figsize = (8, 2))
    ax = fig.add_subplot(111)
    
    Q = ax.imshow(img, cmap="coolwarm", vmin=vmin, vmax=vmax)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    
    height = img.shape[0]
    width = img.shape[1]
    
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_xticks(np.linspace(0, width-41, split_x+1))
    ax.set_yticks(np.linspace(2, height-1, split_y+1))
    ax.set_xticklabels(np.round(np.linspace(0, 2.2, split_x+1),3))
    ax.set_yticklabels(np.round(np.linspace(0, 0.4, split_y+1),3)[::-1])

    ax.set_aspect("equal")
    
    fig.colorbar(Q, label=label)
    
    #plt.show()
    
    return fig


##############################
       ### Main
##############################

### resentative Value

rep_length = 0.1
rep_velocity = 1.0
rep_viscosity = 0.001

Re = (rep_length * rep_velocity) / rep_viscosity

### physics config

x_range = [0.0, 22.0]
y_range = [0.0, 4.1]
cylinder_center = [2.0, 2.0]
cylinder_radius = 0.5
t_range = [0.0, 10.0]
u_max = 1.5

num_window = 10

### Number of points

"""num_sampling_Xedge = [0, 400]
num_sampling_Yedge = [0, 900]
num_sampling_cylinder = [0, 128]
num_sampling_region = [2728, 2736]"""
"""num_sampling_Xedge = [40, 400]
num_sampling_Yedge = [100, 1000]
num_sampling_cylinder = [20, 200]
num_sampling_region = [412, 4480]"""
num_sampling_Xedge = [0, 2048]
num_sampling_Yedge = [0, 2048]
num_sampling_cylinder = [0, 512]
num_sampling_region = [2048, 4096]

### other config
chunks = 16

### CFD data
CFDdata = np.load("data/ns_unsteady.npy", allow_pickle=True)

coords_cfd = CFDdata.tolist()["coords"]/rep_length
u_cfd = CFDdata.tolist()["u"][-1].reshape(-1,1)
v_cfd = CFDdata.tolist()["v"][-1].reshape(-1,1)
p_cfd = CFDdata.tolist()["p"][-1].reshape(-1,1)
velocity_cfd = np.concatenate([u_cfd, v_cfd, p_cfd], axis=1)

### run
for seq in range(0, num_window):
    
    print("===============================")
    print("           Window:{}".format(seq+1))
    print("===============================")
    
    window_t_range = [0, (t_range[1]+(t_range[1]-t_range[0])*0.1)/num_window]

    ### Generator
    if seq==0:
        generator = BatachDataGenerator_Cylinder(
                x_range,
                y_range,
                cylinder_center,
                cylinder_radius,
                window_t_range,
                u_max,
                num_sampling_Xedge,
                num_sampling_Yedge,
                num_sampling_cylinder,
                num_sampling_region,
                chunks=chunks,
                initial_coord_CFD=coords_cfd,
                initial_u_CFD=velocity_cfd
            )
        
    else:
        generator = BatachDataGenerator_Cylinder(
                x_range,
                y_range,
                cylinder_center,
                cylinder_radius,
                window_t_range,
                u_max,
                num_sampling_Xedge,
                num_sampling_Yedge,
                num_sampling_cylinder,
                num_sampling_region,
                chunks=chunks
                )
        
    print(generator.num_sampling())
        
    input_dim = 3
    output_dim = 3
    num_hidden_layers = 5
    num_neurons_per_layer = 256
    activation_function = "tanh"
    
    ms_dnn = ModifiedMLP(input_dim, output_dim, num_hidden_layers, num_neurons_per_layer, activation_function, s_mean=1.0, s_std=0.1)
    
    if seq==0:
        pinns = PhysicsInformedNN_ExpertGuide(device, ms_dnn,Re, generator)
    else:
        with open("/mnt/MachineLearning/flow_field_prediction/PINNs/case_cylinder_retry/vertificate_ExpertGuide/window_{}/PINNs.pkl".format(seq), mode="br") as f:
            previous_model = pickle.load(f)
        pinns = PhysicsInformedNN_ExpertGuide(device, ms_dnn,Re, generator, previous_model.ms_dnn)

    pinns.set_train_visualizer(
                            output_dir="/mnt/MachineLearning/flow_field_prediction/PINNs/case_cylinder_retry/vertificate_ExpertGuide/window_{}".format(seq+1), 
                            x_range=x_range, 
                            y_range=y_range, 
                            t_range=window_t_range,
                            resolution=[440, 81], 
                            cylinder_center=cylinder_center,
                            cylinder_radius=cylinder_radius,
                            min_max = [0, 2.0], 
                            label="velocity_mag"
                            )
    
    pinns.training_config(num_per_chunks=num_sampling_region[1]//chunks, chunks=chunks, epsilon=1.0, alpha=0.95, f=1000)
    
    ### training
    pinns.train(200000)
    
    ### save_model
    with open("/mnt/MachineLearning/flow_field_prediction/PINNs/case_cylinder_retry/vertificate_ExpertGuide/window_{}/PINNs.pkl".format(seq+1), "wb") as f:
        pickle.dump(pinns, f)
        
    print("\n")


