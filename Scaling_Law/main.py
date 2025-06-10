from data_loader import DataLoader
from fitting import ScalingLawFitter
from evaluation import ScalingLawEvaluator

def main():
    # Initialize components
    train_data_loader = DataLoader(max_steps=34000)
    test_data_loader = DataLoader(max_steps=34000)
    fitter = ScalingLawFitter()
    evaluator = ScalingLawEvaluator(train_data_loader, test_data_loader, fitter)

    # Load data
    train_data_loader.load_data_from_csv(
        csv_file='data/100M_gpt_D_cosine_rope.csv'
    )
    test_data_loader.load_data_from_csv(
        csv_file='data/100M_gpt_D_wsd_rope.csv'
    )
    
    # Fit the model
    best_params, train_r2 = fitter.fit(train_data_loader)

    # Print results
    print(f'Best parameters (L0, A, C, alpha) = {best_params}')
    print(f'Training R^2 = {train_r2}')

    # Evaluate and plot for each test set
    test_r2 = evaluator.evaluate_test_data(best_params)
    print(f'Test R^2 = {test_r2}')
        
    # Plot results
    evaluator.plot_results(
        best_params, 
        train_r2,
        save_path='fit.pdf'
    )

if __name__ == "__main__":
    main()