"""  
made by Kataoka @2023/12/XX.  

"""  

import numpy as np
import torch
import torch.nn as nn


class DNN(nn.Module):

    def __init__(self, input_dim, output_dim, num_layer, num_neuron, activation):
        
        super(DNN, self).__init__()

        # instance variables
        self.input_size = input_dim
        self.output_size = output_dim
        self.num_layer = num_layer
        self.num_neuron = num_neuron

        # layers
        self.inputs = nn.Linear(input_dim, num_neuron)
        for i in range(1, self.num_layer):
            exec_command = "self.fc" + str(i) + " = nn.Linear(num_neuron, num_neuron)"
            exec(exec_command)
        self.outputs = nn.Linear(num_neuron, output_dim)

        # activation
        self.set_activation(activation)

        # weight initialization
        self.initialize_model(activation)
        
        return
    

    def set_activation(self, activation):
        
        # need to custimize
        if activation=="tanh":
            self.activation = nn.Tanh()
        elif activation=="silu":
            self.activation = nn.SiLU()
        else:
            # exception handling
            pass

        return
    

    def initialize_model(self, activation):

        # need to custimize
        if activation=="tanh":

            nn.init.xavier_normal_(self.inputs.weight.data, gain=1.0)
            nn.init.zeros_(self.inputs.bias.data)
            nn.init.xavier_normal_(self.outputs.weight.data, gain=1.0)
            nn.init.zeros_(self.outputs.bias.data)

            for i in range(1, self.num_layer):
                exec_command_1 = "nn.init.xavier_normal_(self.fc" + str(i) + ".weight.data, gain=1.0)"
                exec(exec_command_1)
                exec_command_2 = "nn.init.zeros_(self.fc" + str(i) + ".bias.data)"
                exec(exec_command_2)

        elif activation=="silu":
            
            nn.init.kaiming_normal_(self.inputs.weight.data, gain=1.0)
            nn.init.zeros_(self.inputs.bias.data)
            nn.init.kaiming_normal_(self.outputs.weight.data, gain=1.0)
            nn.init.zeros_(self.outputs.bias.data)

            for i in range(1, self.num_layer):
                exec_command_1 = "nn.init.kaiming_normal_(self.fc" + str(i) + ".weight.data, gain=1.0)"
                exec(exec_command_1)
                exec_command_2 = "nn.init.zeros_(self.fc" + str(i) + ".bias.data)"
                exec(exec_command_2)

        else:
            # exception handling
            pass

        return

        
    def forward(self, x):
            
        x = self.activation(self.inputs(x))
        for i in range(1, self.num_layer):
            exec_command = "x = self.activation(self.fc" + str(i) + "(x))"
            exec(exec_command)
        x = self.outputs(x)
        
        return x
    

class PhysicsInformedNN():

    def __init__(self, input_dim, output_dim, num_layer, num_neuron, activation, optimizer, X, U, Re, device):
        
        # device
        self.device = device

        # Data
        self.X = X
        self.U = U
        self.num_batch = len(X["residual"])
        
        # Reynolds Number
        self.Re = Re
        
        # Loss History
        self.loss_history = {"initial": [], "boundary": [], "residual": [], "total": []}
        
        # Deep Neural Networks
        self.dnn = DNN(input_dim, output_dim, num_layer, num_neuron, activation).to(self.device)
        
        # optimizer
        self.optimize_method = optimizer
        self.set_optimizer(optimizer)
        
        # epoch
        self.epoch = 0

        # loss weight
        self.alpha = 10.0     # weight for initial loss
        self.beta = 10.0      # weight for boundary loss
        self.gamma = 1.0     # weight for residual loss

        # setting
        self.display_config = 2
        self.history_config = 1
        self.output_freq = 10
        self.record_freq = 10


    def set_optimizer(self, optimizer, params={}):

        if optimizer=="adam":
            self.optimizer = torch.optim.Adam(self.dnn.parameters(), **params)
            self.optimize_closure = False
            return
        
        elif optimizer=="lbfgs":
            self.optimizer = torch.optim.LBFGS(self.dnn.parameters(), **params)
            self.optimize_closure = True
            return
    

    def set_optimizer_params(self, params):

        self.set_optimizer(self.optimize_method, params)

        return

    
    def net_u(self, x, y, t):  
        
        outputs = self.dnn(torch.cat([x, y, t], dim=1).to(self.device))

        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        return u, v, p


    def net_u_t(self, x, y, t):  
        
        u, v, _ = self.net_u(x, y, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        return u_t, v_t
    

    def net_u_x(self, x, y, t):  
        
        u, v, p = self.net_u(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        return u_x, v_x, p_x


    def net_u_y(self, x, y, t):  
        
        u, v, p = self.net_u(x, y, t)

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        return u_y, v_y, p_y
    

    def net_u_xx(self, x, y, t):  
        
        u, v, _ = self.net_u(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]

        return u_xx, v_xx


    def net_u_yy(self, x, y, t):  
        
        u, v, _ = self.net_u(x, y, t)

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        return u_yy, v_yy
    

    def net_f(self, x, y, t):

        u, v, _ = self.net_u(x, y, t)
        
        u_t, v_t = self.net_u_t(x, y, t)
        u_x, v_x, p_x = self.net_u_x(x, y, t)
        u_y, v_y, p_y = self.net_u_y(x, y, t)
        u_xx, v_xx = self.net_u_xx(x, y, t)
        u_yy, v_yy = self.net_u_yy(x, y, t)
        
        f_x = u_t + u * u_x + v * u_y + p_x - 1/self.Re * (u_xx + u_yy)
        f_y = v_t + u * v_x + v * v_y + p_y - 1/self.Re * (v_xx + v_yy)
        f_continue = u_x + v_y
        
        return f_x, f_y, f_continue
    

    def culc_loss(self, batch, x=None, y=None, t=None):

        # initial_loss
        keys = [key for key in list(self.U.keys()) if self.U[key].point_type=="initial"]

        num_data = 0
        init_loss = 0

        for key in keys:
            x_init = torch.tensor(self.X[key][batch][:, 0:1], requires_grad=True).float().to(self.device)
            y_init = torch.tensor(self.X[key][batch][:, 1:2], requires_grad=True).float().to(self.device)
            t_init = torch.tensor(self.X[key][batch][:, 2:3], requires_grad=True).float().to(self.device)

            u_init = torch.tensor(self.U[key].values()[batch][:, 0:1]).float().to(self.device)
            v_init = torch.tensor(self.U[key].values()[batch][:, 1:2]).float().to(self.device)

            init_loss += self.loss(self.U[key].condition, x_init, y_init, t_init, u_init, v_init)

            num_data += len(x_init)

        init_loss = init_loss/num_data
        
        del x_init, y_init, t_init, u_init, v_init
        torch.cuda.empty_cache()

        # boundary_loss
        keys = [key for key in list(self.U.keys()) if self.U[key].point_type=="boundary"]

        num_data = 0
        bound_loss = 0

        for key in keys:
            x_bound = torch.tensor(self.X[key][batch][:, 0:1], requires_grad=True).float().to(self.device)
            y_bound = torch.tensor(self.X[key][batch][:, 1:2], requires_grad=True).float().to(self.device)
            t_bound = torch.tensor(self.X[key][batch][:, 2:3], requires_grad=True).float().to(self.device)

            u_bound = torch.tensor(self.U[key].values()[batch][:, 0:1]).float().to(self.device)
            v_bound = torch.tensor(self.U[key].values()[batch][:, 1:2]).float().to(self.device)

            bound_loss += self.loss(self.U[key].condition, x_bound, y_bound, t_bound, u_bound, v_bound)

            num_data += len(x_bound)

        bound_loss = bound_loss/num_data

        del x_bound, y_bound, t_bound, u_bound, v_bound
        torch.cuda.empty_cache()

        # residual_loss
        keys = [key for key in list(self.U.keys()) if self.U[key].point_type=="residual"]

        num_data = 0
        res_loss = 0

        for key in keys:
            x_res = torch.tensor(self.X[key][batch][:, 0:1], requires_grad=True).float().to(self.device)
            y_res = torch.tensor(self.X[key][batch][:, 1:2], requires_grad=True).float().to(self.device)
            t_res = torch.tensor(self.X[key][batch][:, 2:3], requires_grad=True).float().to(self.device)

            f_x, f_y, f_continue = self.net_f(x_res, y_res, t_res)
            res_loss += torch.sum(f_x**2 + f_y**2 + f_continue**2)

            num_data += len(x_res)

        res_loss = res_loss/num_data

        # total_loss
        total_loss = self.alpha*init_loss + self.beta*bound_loss + self.gamma*res_loss

        del x_res, y_res, t_res
        torch.cuda.empty_cache()

        return init_loss, bound_loss, res_loss, total_loss


    def loss(self, condition, x, y, t, u, v):

        if condition=="dirichlet":
            loss = self.loss_dirichlet(x, y, t, u, v)
        elif condition=="neumann_x":
            loss = self.loss_neumann(x, y, t, u, v, direction="x")
        elif condition=="neumann_y":
            loss = self.loss_neumann(x, y, t, u, v, direction="y")
        else:
            pass    # exception handling

        return loss


    def loss_dirichlet(self, x, y, t, u, v):

        u_pred, v_pred, _ = self.net_u(x, y, t)
        init_loss = torch.sum((u - u_pred) ** 2 + (v - v_pred) ** 2)

        return init_loss
    

    def loss_neumann(self, x, y, t, u, v, direction):

        if direction=="x":
            u_pred, v_pred, _ = self.net_u_x(x, y ,t)
        elif direction=="y":
            u_pred, v_pred, _ = self.net_u_y(x, y ,t)
        else:
            pass    # exception handling

        init_loss = torch.sum((u - u_pred) ** 2 + (v - v_pred) ** 2)

        return init_loss
        

    def train(self, nIter):
        
        # Training
        self.dnn.train()

        for epoch in range(1, nIter+1):

            self.epoch += 1

            for batch in range(self.num_batch):

                if self.optimize_closure==True:

                    def closure():
                        self.optimizer.zero_grad()
                    
                        _, _, _, loss = self.culc_loss(batch=batch)

                        loss.backward()
                        
                        return loss
                    
                    self.optimizer.step(closure)

                    init_loss, bound_loss, res_loss, loss = self.culc_loss(batch=batch)

                else:
                    init_loss, bound_loss, res_loss, loss = self.culc_loss(batch=batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if batch<self.num_batch-1:
                    self.update_teacher_data(batch)

                # if self.display_config==2 and epoch%self.output_freq==0:

                #     print(
                #         "    init_Loss: %.6f, boundary_Loss: %.6f, residual_Loss: %.6f, total_loss: %.6f" % 
                #         (
                #             init_loss.item(),
                #             bound_loss.item(), 
                #             res_loss.item(), 
                #             loss.item()
                #         )
                #     )

            #init_loss, bound_loss, res_loss, loss = self.culc_loss()

            if self.epoch%self.output_freq==0:
                print(
                    "Epoch: %d, init_Loss: %.6f, boundary_Loss: %.6f, residual_Loss: %.6f, total_loss: %.6f" % 
                    (
                        self.epoch,
                        init_loss.item(),
                        bound_loss.item(), 
                        res_loss.item(), 
                        loss.item()
                    )
                )

            if self.epoch%self.record_freq==0:
                self.loss_history["initial"].append(init_loss.item())
                self.loss_history["boundary"].append(bound_loss.item())
                self.loss_history["residual"].append(res_loss.item())
                self.loss_history["total"].append(loss.item())


    def update_teacher_data(self, batch):

        keys = [key for key in list(self.U.keys()) if self.U[key].point_type=="initial"]

        for key in keys:

            u, v, _ = self.predict(self.X[key][batch+1])

            u = u.reshape(-1,1)
            v = v.reshape(-1,1)

            self.U[key].values()[batch+1] = np.concatenate([u,v], axis=1)

        return


    def predict(self, X):

        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        
        self.dnn.eval()
        u, v, p = self.net_u(x, y, t)

        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        
        return u, v, p