import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, max_steps=20000, decay_factor=0.999):
        """Initialize the data loader.
        
        Args:
            max_steps (int): Maximum number of steps
            decay_factor (float): Decay factor for momentum calculation
        """
        self.max_steps = max_steps
        self.decay_factor = decay_factor
        self.steps = []
        self.losses = []
        self.lr = []
        self.S1 = []
        self.momentum = []
        self.S2 = []
        
    def load_data_from_csv(self, csv_file):
        """Load training and test data from CSV files.
        
        Args:
            train_file (str): Path to training data CSV file
            test_files (list): List of paths to test data CSV files
        """
        # Load data
        df = pd.read_csv(csv_file)
        self.steps = np.array(df['step'].values)
        self.losses = np.array(df['loss'].values)
        self.lr = np.array(df['lr'].values)
        print(len(self.steps), len(self.losses), len(self.lr))

        self.S1 = np.cumsum(self.lr)
        n = len(self.lr)
        self.momentum = np.zeros(n)
        for i in range(1, n):
            self.momentum[i] = (
                self.decay_factor * self.momentum[i-1] + 
                (self.lr[i-1] - self.lr[i])
            )
        self.S2 = np.cumsum(self.momentum)
            
    def get_data(self):
        """Get the training data.
        
        Returns:
            tuple: (steps, losses, lr, S1, S2)
        """
        return (
            self.steps,
            self.losses,
            self.lr,
            self.S1,
            self.S2
        )
        