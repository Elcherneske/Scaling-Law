import math
import numpy as np

# def lr(current_step, max_lr=2e-4, min_lr=2e-5, total_steps=60000, warmup_steps=500, lr_method='cosine'):
#     """Learning rate scheduler with different methods.
    
#     Args:
#         current_step (int): Current training step
#         max_lr (float): Maximum learning rate
#         min_lr (float): Minimum learning rate
#         total_steps (int): Total training steps
#         warmup_steps (int): Number of warmup steps
#         lr_method (str): Learning rate schedule method ('constant', 'linear', or 'cosine')
    
#     Returns:
#         float: Current learning rate
#     """
#     if lr_method == 'constant':
#         return max_lr

#     if current_step <= warmup_steps:
#         return max_lr

#     num_steps_ = current_step - warmup_steps
#     annealing_steps_ = total_steps - warmup_steps
#     delta_lr = max_lr - min_lr
    
#     if lr_method == 'linear':
#         decay_ratio = float(num_steps_) / float(annealing_steps_)
#         coeff = (1.0 - decay_ratio)
#         current_lr = min_lr + coeff * delta_lr
#     elif lr_method == 'cosine':
#         decay_ratio = float(num_steps_) / float(annealing_steps_)
#         coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
#         current_lr = min_lr + coeff * delta_lr
#     else:
#         raise Exception('{} decay style is not supported.'.format(lr_method))

#     return current_lr

def huber_loss(residual, delta):
    """Huber loss function for robust fitting.
    
    Args:
        residual (np.ndarray): Residual values
        delta (float): Delta parameter for Huber loss
    
    Returns:
        np.ndarray: Huber loss values
    """
    return np.where(np.abs(residual) < delta, 
                   0.5*((residual)**2), 
                   delta*np.abs(residual) - 0.5*(delta**2)) 