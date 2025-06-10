import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

class ScalingLawEvaluator:
    def __init__(self, train_data_loader, test_data_loader, fitter):
        """Initialize the evaluator.
        
        Args:
            data_loader (DataLoader): DataLoader instance
            fitter (ScalingLawFitter): ScalingLawFitter instance
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.fitter = fitter
        
    def evaluate_test_data(self, best_params, test_idx=0):
        """Evaluate the model on test data.
        
        Args:
            best_params (tuple): Best parameters from fitting
            test_idx (int): Index of the test set to evaluate
            
        Returns:
            float: R2 score on test data
        """
        steps, losses, lr, S1, S2 = self.test_data_loader.get_data()
        if steps is None:
            return None
            
        predict_losses = []
        for step in steps:
            predict_loss = self.fitter.Howe_Scaling_Law(
                step, *best_params,
                S1, S2
            )
            predict_losses.append(predict_loss)
            
        predict_losses = np.array(predict_losses).astype(np.float32)
        ss_res = np.sum((np.log(losses) - np.log(predict_losses)) ** 2)
        ss_tot = np.sum((np.log(losses) - np.mean(np.log(losses))) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
        
    def plot_results(self, best_params, r2, test_idx=None, save_path="fit.pdf"):
        """Plot the fitting results.
        
        Args:
            best_params (tuple): Best parameters from fitting
            r2 (float): R2 score on training data
            test_idx (int): Index of test set to plot (if None, only plot training data)
            save_path (str): Path to save the plot
        """
        steps, losses, lr, S1, S2 = self.train_data_loader.get_data()
        L0, A, C, alpha = best_params
        
        plt.figure(figsize=(12, 8))
        
        # Sample data points at regular intervals
        num_points = 100  # Number of points to plot
        indices = np.linspace(0, len(steps)-1, num_points, dtype=int)
        sampled_steps = steps[indices]
        sampled_losses = losses[indices]
        
        # Plot training data
        predict_losses = []
        for step in sampled_steps:
            predict_losses.append(
                self.fitter.Howe_Scaling_Law(
                    step, *best_params, S1, S2
                )
            )
        predict_losses = np.array(predict_losses).astype(np.float32)
        
        plt.plot(sampled_steps, sampled_losses, 'x', 
                markersize=4, label=f'Train Loss', 
                color=f"C1")
        plt.plot(sampled_steps, predict_losses, '--', 
                label=f'Fitting Curve', 
                color=f"C0", zorder=3)
            
        plt.yticks(np.arange(2.8, 4.0, 0.1))
        plt.ylim(2.8, 4.0)
        plt.xlim(0, self.train_data_loader.max_steps)
        
        # Add text annotations
        plt.text(0.2, 0.5, 
                f'Fitting Curve: L = {L0:.3f} + {A:.3f}*S1^(-{alpha:.3f}) - {C:.3f}*S2', 
                fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.2, 0.4, r'Train R^2 = ' + f'{r2:.5f}', fontsize=10, transform=plt.gca().transAxes)

        
        plt.grid()
        plt.legend(prop=FontProperties(size=12))
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close() 