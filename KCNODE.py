from torchdyn.core import NeuralODE
import torch


class BaselineNN(torch.nn.Module):
    def __init__(self, n_hidden=8):
        super(BaselineNN, self).__init__()
        self.layer_1 = torch.nn.Linear(6, n_hidden)
        self.layer_3 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer_out = torch.nn.Linear(n_hidden, 5)     
        
    def forward(self, x):
        molar_flows = x[:, :-2]
        temperature = x[:, -2:-1]        
        total_pressure = x[:, -1:]        
        partial_pressure = molar_flows[:, :-1] / molar_flows.sum(axis=1, keepdim=True) * total_pressure        
        
        x = torch.tanh(self.layer_1(torch.hstack([partial_pressure, temperature])))        
        x = torch.exp(self.layer_3(x))
        x = self.layer_out(x)       
        
        return torch.hstack([x, torch.zeros_like(x[:, :3])])  
  
    
class KineticConstrainedNN(torch.nn.Module):
    def __init__(self, n_hidden=8):
        super(KineticConstrainedNN, self).__init__()
        self.layer_1 = torch.nn.Linear(5, n_hidden)        
        self.layer_T = torch.nn.Linear(1, 2)        
        self.layer_out = torch.nn.Linear(n_hidden, 2)
        
        self.coef_ = torch.tensor([[-1, 1, 0], # the stoichiometric matrix 
                                   [-1, 1,-3],
                                   [ 1,-1,-1],
                                   [ 1,-1, 1],
                                   [ 0, 0, 1],
                                   [ 0, 0, 0]]).T.float()        
        
    def forward(self, x):
        molar_flows = x[:, :-2]
        temperature = x[:, -2:-1]
        total_pressure = x[:, -1:]
        
        partial_pressure = molar_flows[:, :-1] / molar_flows.sum(axis=1, keepdim=True) * total_pressure
        
        p_CO2 = partial_pressure[:, 0:1]
        p_H2 = partial_pressure[:, 1:2]
        p_CO = partial_pressure[:, 2:3]
        p_H2O = partial_pressure[:, 3:4]        
        
        x = torch.tanh(self.layer_1(partial_pressure))
        x = torch.sigmoid(self.layer_out(x))

        xt = torch.exp(10*self.layer_T(temperature))
        
        T = 1 / (temperature*8.31/10000 + 1 / (273.15+300))           
        K_eq = torch.exp(3.933 - 4076/(T - 39.64))            
        xt = torch.hstack([xt[:, :1], xt[:, :1] / K_eq, xt[:, 1:]])
        
        x1 = p_CO2 * p_H2 * x[:, 0:1]
        x2 = p_CO * p_H2O * x[:, 0:1]
        x3 = p_CO * p_H2 * x[:, 1:2]
        
        x = torch.hstack([x1, x2, x3])
        x = x * xt
        x = torch.linalg.multi_dot([x, self.coef_])
        
        return torch.hstack([x, torch.zeros_like(x[:, :2])])
    
    
class MyLinear(torch.nn.Linear):
    '''
    To make parameters related to activation energy be positive
    '''
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)  

    def forward(self, input):
        return torch.nn.functional.linear(input, -torch.abs(self.weight), self.bias)


class KineticConstrainedFTNN(torch.nn.Module):
    def __init__(self, n_hidden=8):
        super(KineticConstrainedFTNN, self).__init__()
        self.layer_1 = torch.nn.Linear(6, n_hidden)        
        self.layer_T = MyLinear(1, 5)        
        self.layer_out = torch.nn.Linear(n_hidden, 12)
        
        self.amount_an = 15
        self.amount_en = self.amount_an - 1
        self.amount_oh = 7
        
        self.create_matrix()

        
    def create_matrix(self):
        self.coef_ = torch.tensor([[ 0, 0, 0], # N2
                                   [-1, 1,-1], # CO2
                                   [ 1,-1, 0], # CO
                                   [-1, 1,-4], # H2
                                   [ 1,-1, 2], # H2O  
                                  ])

        self.coef_ = torch.vstack([self.coef_, torch.zeros([self.amount_an + self.amount_en + self.amount_oh, 3])])        
        self.coef_[5, 2] = 1

        M11 = torch.zeros((5, self.amount_an))
        M11[2] = -1 # CO
        M11[3] = -(2 + 1 / torch.arange(1, self.amount_an+1)) # H2        
        M11[4] = +1 # H2O

        M12 = torch.zeros((5, self.amount_en))
        M12[2] = -1 # CO
        M12[3] = -2 # H2        
        M12[4] = +1 # H2O
        
        M13 = torch.zeros((5, self.amount_oh))
        M13[2] = -1 # CO
        M13[3] = -2 # H2                
        M13[4] = +(1 - 1 / torch.arange(1, self.amount_oh+1)) # H2O

        M1 = torch.hstack([M11, M12, M13])

        M21 = torch.eye(self.amount_an) * 1 / torch.arange(1, self.amount_an+1)
        M22 = torch.zeros((self.amount_an, self.amount_en))
        M23 = torch.zeros((self.amount_an, self.amount_oh))
        
        M2 = torch.hstack([M21, M22, M23])

        M31 = torch.zeros((self.amount_en, self.amount_an))
        M32 = torch.eye(self.amount_en) * 1 / torch.arange(2, self.amount_en+2)
        M33 = torch.zeros((self.amount_en, self.amount_oh))
        
        M3 = torch.hstack([M31, M32, M33])
        
        M41 = torch.zeros((self.amount_oh, self.amount_an))
        M42 = torch.zeros((self.amount_oh, self.amount_en))
        M43 = torch.eye(self.amount_oh) * 1 / torch.arange(1, self.amount_oh+1)
        
        M4 = torch.hstack([M41, M42, M43])        

        M = torch.vstack([M1, M2, M3, M4])
        self.coef_ = torch.hstack([self.coef_, M]).T.float()

        
    def forward(self, x):        
        molar_flows = x[:, 3:]
        total_pressure = x[:, 0:1]
        temperature = x[:, 1:2]
        tos = x[:, 2:3]
        
        partial_pressure = molar_flows[:, 1:5] / molar_flows.sum(axis=1, keepdim=True) * total_pressure
        
        p_CO2 = partial_pressure[:, 0:1]
        p_CO = partial_pressure[:, 1:2]
        p_H2 = partial_pressure[:, 2:3]
        p_H2O = partial_pressure[:, 3:4] 
        
        x = torch.hstack([temperature, tos, partial_pressure])
        x = torch.tanh(self.layer_1(x))
        
        x = torch.sigmoid(self.layer_out(x))
        
        # Scaling outputs for empirical parameters        
        alpha1 = x[:, 5:6]*0.3+0.5
        alpha2 = x[:, 6:7]*0.2
        fraction = x[:, 7:8]*0.3
        alpha_en = x[:, 8:9]*0.4+0.3
        frac_en2 = x[:, 9:10]*0.6        
        alpha_oh = x[:, 10:11]*0.5        
        frac_oh1 = x[:, 11:12]

        xt = torch.exp(self.layer_T(temperature))
        T = 1 / (temperature*8.31/10000 + 1 / (273.15+280))
        K_eq = torch.exp(3.933 - 4076/(T - 39.64))
        xt = torch.hstack([xt[:, 0:1], xt[:, 0:1] / K_eq, xt[:, 1:5]])
        
        rwgs_f = p_CO2 * p_H2 * x[:, 0:1]
        rwgs_b = p_CO * p_H2O * x[:, 0:1]
        methanation = p_CO2 * p_H2 * x[:, 1:2]
        ft_an = p_CO * p_H2 * x[:, 2:3]
        ft_en = p_CO * p_H2 * x[:, 3:4]
        ft_oh = p_CO * p_H2 * x[:, 4:5]
        
        x = torch.hstack([rwgs_f, rwgs_b, methanation, ft_an, ft_en, ft_oh])
        rates = x * xt
        
        rate_ft_an = rates[:, -3:-2]
        rate_ft_en = rates[:, -2:-1]
        rate_ft_oh = rates[:, -1:]
        
        carbon_number = torch.arange(1, self.amount_an + 1)
        rate_an = fraction * carbon_number * rate_ft_an * torch.pow(alpha1, carbon_number - 1) + (1-fraction) * carbon_number * rate_ft_an * torch.pow(alpha2, carbon_number - 1)
        
        carbon_number = torch.arange(2, self.amount_en + 2)
        rate_en = carbon_number * rate_ft_en * torch.pow(alpha_en, carbon_number - 1)
        rate_en[:, 0:1] *= frac_en2
        
        carbon_number = torch.arange(1, self.amount_oh + 1)
        rate_oh = carbon_number * rate_ft_oh * torch.pow(alpha_oh, carbon_number - 1)
        rate_oh[:, 0:1] *= frac_oh1
        
        x = torch.hstack([rates[:, :3],  rate_an, rate_en, rate_oh])


        x = torch.linalg.multi_dot([x, self.coef_])
        out = torch.hstack([torch.zeros_like(x[:, :3]), x])
        
        return out
        
    
def get_baseline_model(solver='dopri5'):
    model_nn = BaselineNN()
    return NeuralODE(model_nn, sensitivity='autograd', 
                     solver=solver, order=1, return_t_eval=False)


def get_KCNODE_methanation_model(solver='dopri5'):
    model_nn = KineticConstrainedNN()
    return NeuralODE(model_nn, sensitivity='autograd', 
                     solver=solver, order=1, return_t_eval=False)


def get_KCNODE_FT_model(solver='dopri5'):
    model_nn = KineticConstrainedFTNN()
    return NeuralODE(model_nn, sensitivity='autograd', 
                     solver=solver, order=1, return_t_eval=False)
    
