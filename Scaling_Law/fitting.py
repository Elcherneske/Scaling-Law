import numpy as np
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm
from utils import huber_loss

class ScalingLawFitter:
    def __init__(self, huber_delta=1e-3):
        """Initialize the scaling law fitter.
        
        Args:
            huber_delta (float): Delta parameter for Huber loss
        """
        self.huber_delta = huber_delta
        
    def Howe_Scaling_Law(self, step, L0, A, C, alpha, S1, S2):
        """Howe's scaling law model.
        
        Args:
            step (int): Current step
            L0 (float): Initial loss
            A (float): Scaling coefficient
            C (float): Momentum coefficient
            alpha (float): Power law exponent
            S1 (np.ndarray): S1 metrics
            S2 (np.ndarray): S2 metrics
            
        Returns:
            float: Predicted loss
        """
        predict_loss = L0 + A*(1/S1[step])**alpha - C*S2[step]
        return predict_loss
        
    def objective(self, params, fitting_steps, fitting_losses, S1, S2):
        """Objective function for optimization.
        
        Args:
            params (tuple): Model parameters (L0, A, C, alpha)
            fitting_steps (np.ndarray): Training steps
            fitting_losses (np.ndarray): Training losses
            S1 (np.ndarray): S1 metrics
            S2 (np.ndarray): S2 metrics
            
        Returns:
            float: Total loss
        """
        L0, A, C, alpha = params
        loss = 0
        
        predict_losses = self.Howe_Scaling_Law(
            fitting_steps, 
            L0, A, C, alpha,
            S1, S2
        )
            
        residual = np.log(fitting_losses) - np.log(predict_losses)
        loss += huber_loss(residual, self.huber_delta).sum()
            
        return loss
        
    def fit(self, data_loader, init_ranges=None):
        """Fit the scaling law model.
        
        Args:
            data_loader (DataLoader): DataLoader instance
            init_ranges (dict): Initial parameter ranges for grid search
            
        Returns:
            tuple: Best parameters and R2 score
        """
        if init_ranges is None:
            init_ranges = {
                'L0': np.linspace(0.1, 2.1, 2),
                'A': np.linspace(1, 22, 3),
                'C': np.linspace(1, 22, 3),
                'alpha': np.linspace(0, 0.8, 3)
            }
            
        fitting_steps, fitting_losses, fitting_lr, S1, S2 = data_loader.get_data()
        
        best_params = None
        best_loss = np.inf
        
        initial_params = product(
            init_ranges['L0'],
            init_ranges['A'],
            init_ranges['C'],
            init_ranges['alpha']
        )
        
        for initial_param in tqdm(initial_params):
            result = minimize(
                lambda x: self.objective(x, fitting_steps, fitting_losses, S1, S2),
                initial_param,
                method='L-BFGS-B',
                bounds=[(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)],
                options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'eps': 1e-8}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
                
        # Compute R2
        predict_losses = []
        for fitting_step in fitting_steps:
            predict_loss = self.Howe_Scaling_Law(
                fitting_step, *best_params, S1, S2
            )
            predict_losses.append(predict_loss)
            
        predict_losses = np.array(predict_losses).astype(np.float32)
        ss_res = np.sum((np.log(fitting_losses) - np.log(predict_losses)) ** 2)
        ss_tot = np.sum((np.log(fitting_losses) - np.mean(np.log(fitting_losses))) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return best_params, r2 