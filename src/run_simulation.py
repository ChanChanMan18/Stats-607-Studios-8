"""
Main simulation runner for generating simulation results CSV.

This script integrates generate_data.py and huber_reg.py to create simulation
results in the format required by evaluation.py.

Output CSV format: method, n, p, df, rho, SNR, rep, mse
"""

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Import from existing modules
from generate_data import simulate_dataset
from run_reg import ridgeless_beta_hat, calculate_mse


def run_single_simulation(n, gamma, seed):
    """
    Run a single simulation: generate data, fit model, calculate MSE.
    
    Parameters
    ----------
    n : int
        Number of samples
    gamma : float
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing experimental and theoretical MSE
    """
    # Generate data using the provided function
    data = simulate_dataset(gamma=gamma, n = 200, seed=seed)
    
    X = data["X"]
    y = data["y"]
    beta_true = data["beta"]
    params = data["params"]
    
    # Run ridgeless regression using ridgeless_beta_hat from run_reg.py
    # No intercept is added, so dimensions match perfectly
    beta_hat = ridgeless_beta_hat(X, y)
    
    # Use calculate_mse from run_reg.py
    # This returns (experimental_mse, theoretical_mse)
    experimental_mse, theoretical_mse = calculate_mse(
        beta_hat=beta_hat,
        beta_true=beta_true,
        gamma=gamma,
        sigma_squared=params['sigma']**2,
        r_squared=params['r']**2
    )
    
    return {
        'experimental_mse': experimental_mse,
        'theoretical_mse': theoretical_mse
    }


def run_simulation_regime(n, gamma_values, n_sim, regime_name):
    """
    Run all simulations for a specific regime (fixed n, different gamma, multiple replications).
    
    Parameters
    ----------
    n : int
        Fixed sample size per experiment
    gamma_values : array-like
        Gamma ratios to test
    n_sim : int
        Number of replications per gamma value
    regime_name : str
        Name of the regime (for display)
    """
    results = []
    print(f"\n{regime_name}: n = {n}, n_sim = {n_sim}, {len(gamma_values)} γ values")

    for gamma in tqdm(gamma_values, desc="  Running γ values"):
        mse_values = []
        theoretical_mse = None

        for rep in range(n_sim):
            seed = np.random.randint(0, 1_000_000)
            result = run_single_simulation(n, gamma, seed)
            mse_values.append(result["experimental_mse"])
            if theoretical_mse is None:
                theoretical_mse = result["theoretical_mse"]

        mean_mse = np.mean(mse_values)
        se_mse = 0.0 if n_sim == 1 else np.std(mse_values, ddof=1) / np.sqrt(n_sim)

        results.append({
            "n_sim": n_sim,
            "gamma": gamma,
            "MSE": mean_mse,
            "se(MSE)": se_mse #,
            #"theoretical_MSE": theoretical_mse
        })
    return results




def main():
    n = 200  # Fixed number of samples per experiment

    # Define regimes: keep total ~5000 runs
    regimes = [
        {
            "n_sim": 1,
            "gamma_values": np.logspace(np.log10(0.1), np.log10(10), 5000),
            "name": "Regime 1: n_sim = 1"
        },
        {
            "n_sim": 50,
            "gamma_values": np.logspace(np.log10(0.1), np.log10(10), 100),
            "name": "Regime 2: n_sim = 50"
        },
        {
            "n_sim": 1000,
            "gamma_values": np.array([0.2, 0.5, 0.8, 2, 5]),
            "name": "Regime 3: n_sim = 1000"
        }
    ]

    all_results = []

    for reg in regimes:
        res = run_simulation_regime(
            n=n,
            gamma_values=reg["gamma_values"],
            n_sim=reg["n_sim"],
            regime_name=reg["name"]
        )
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    df.to_csv("simulation_results.csv", index=False)
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()




































#
#import numpy as np
#import pandas as pd
#import os
#from itertools import product
#from tqdm import tqdm
#
## Import from existing modules
#from generate_data import simulate_dataset
#from run_reg import lin_regression, huber_regression, quantile_regression, calculate_mse
#
#
## def calculate_mse(beta_hat, beta_true):
##     """Calculate Mean Squared Error between estimated and true coefficients"""
##     return np.sum((beta_hat - beta_true) ** 2)
#
#
#def run_single_simulation(n, gamma, rho, df, snr, method, rep_id, seed=None):
#    """
#    Run a single simulation with specified parameters
#    
#    Parameters:
#    -----------
#    n : int
#        Sample size
#    gamma : float
#        Aspect ratio (p/n)
#    rho : float
#        AR(1) correlation parameter
#    df : float
#        Degrees of freedom for t-distribution (np.inf for normal)
#    snr : float
#        Signal-to-noise ratio
#    method : str
#        Regression method ('OLS', 'LAD', 'Huber')
#    rep_id : int
#        Replication number
#    seed : int, optional
#        Random seed for reproducibility
#    
#    Returns:
#    --------
#    dict
#        Simulation result with all parameters and MSE
#    """
#    
#    # Generate simulation data
#    sim_data = simulate_dataset(n=n, gamma=gamma, rho=rho, df=df, snr=snr, seed=seed)
#    
#    X = sim_data['X']
#    y = sim_data['y']
#    beta_true = sim_data['beta']
#    p = sim_data['params']['p']
#    
#    # Fit model based on method
#    try:
#        if method == 'OLS':
#            beta_hat = lin_regression(X, y)
#        elif method == 'LAD':
#            beta_hat = quantile_regression(X, y, alpha=0.5)  # Median regression
#        elif method == 'Huber':
#            beta_hat = huber_regression(X, y, epsilon=1.35)
#        else:
#            raise ValueError(f"Unknown method: {method}")
#        
#        # Calculate MSE
#        mse = calculate_mse(beta_hat, beta_true)
#        
#    except Exception as e:
#        print(f"Warning: Simulation failed for method {method}, rep {rep_id}: {e}")
#        mse = np.nan
#    
#    return {
#        'method': method,
#        'n': n,
#        'p': p,
#        'df': df if not np.isinf(df) else 1000,  # Use large number for infinity
#        'rho': rho,
#        'SNR': snr,
#        'rep': rep_id,
#        'mse': mse
#    }
#
#
#def run_simulation_study(
#    n_values=[100, 200],
#    gamma_values=[0.1, 0.2, 0.5],  # p/n ratios
#    rho_values=[0.0, 0.3, 0.9],
#    df_values=[1, 3, 10],
#    snr_values=[0.5, 1.0, 5.0],
#    methods=['OLS', 'LAD', 'Huber'],
#    n_reps=50,
#    output_file='artifacts/simulation_results.csv',
#    random_seed=42
#):
#    """
#    Run complete simulation study
#    
#    Parameters:
#    -----------
#    n_values : list
#        Sample sizes to test
#    gamma_values : list
#        Aspect ratios (p/n) to test
#    rho_values : list
#        AR(1) correlation parameters
#    df_values : list
#        Degrees of freedom for t-distribution
#    snr_values : list
#        Signal-to-noise ratios
#    methods : list
#        Regression methods to compare
#    n_reps : int
#        Number of replications per scenario
#    output_file : str
#        Path to save results CSV
#    random_seed : int
#        Base random seed
#    
#    Returns:
#    --------
#    pd.DataFrame
#        Complete simulation results
#    """
#    
#    print("="*60)
#    print("SIMULATION STUDY CONFIGURATION")
#    print("="*60)
#    print(f"Sample sizes (n): {n_values}")
#    print(f"Aspect ratios (γ): {gamma_values}")
#    print(f"Correlations (ρ): {rho_values}")
#    print(f"Degrees of freedom: {df_values}")
#    print(f"SNR values: {snr_values}")
#    print(f"Methods: {methods}")
#    print(f"Replications per scenario: {n_reps}")
#    print(f"Output file: {output_file}")
#    print("="*60)
#    
#    # Generate all parameter combinations
#    param_combinations = list(product(
#        n_values, gamma_values, rho_values, df_values, snr_values, methods
#    ))
#    
#    total_sims = len(param_combinations) * n_reps
#    print(f"Total simulations to run: {total_sims:,}")
#    print("="*60)
#    
#    # Initialize results list
#    results = []
#    
#    # Set up random number generator
#    rng = np.random.default_rng(random_seed)
#    
#    # Run simulations with progress bar
#    with tqdm(total=total_sims, desc="Running simulations") as pbar:
#        for n, gamma, rho, df, snr, method in param_combinations:
#            for rep in range(n_reps):
#                # Generate unique seed for each simulation
#                sim_seed = rng.integers(0, 2**31)
#                
#                # Run single simulation
#                result = run_single_simulation(
#                    n=n, gamma=gamma, rho=rho, df=df, snr=snr,
#                    method=method, rep_id=rep, seed=sim_seed
#                )
#                
#                results.append(result)
#                pbar.update(1)
#    
#    # Convert to DataFrame
#    df_results = pd.DataFrame(results)
#    
#    # Remove rows with NaN MSE values
#    n_failed = df_results['mse'].isna().sum()
#    if n_failed > 0:
#        print(f"\nWarning: {n_failed} simulations failed and were removed")
#        df_results = df_results.dropna(subset=['mse'])
#    
#    # Create output directory if needed
#    os.makedirs(os.path.dirname(output_file), exist_ok=True)
#    
#    # Save results
#    df_results.to_csv(output_file, index=False)
#    print(f"\nResults saved to: {output_file}")
#    print(f"Final dataset shape: {df_results.shape}")
#    
#    return df_results
#
#
#def run_full_simulation(output_file='artifacts/full_simulation_results.csv'):
#    """
#    Run the complete simulation study
#    """
#    print("Running full simulation study...")
#    
#    return run_simulation_study(
#        n_values=[50, 100, 200],
#        gamma_values=[0.2, 0.5],
#        rho_values=[ 0.5, 0.8],
#        df_values=[1, 5, 10],
#        snr_values=[1.0, 3.0, 5.0],
#        methods=['OLS', 'LAD', 'Huber'],
#        n_reps=20,
#        output_file=output_file,
#        random_seed=42
#    )
#
#
#if __name__ == "__main__":
#    # Run the simulation study
#    print("Running simulation study...")
#    df = run_full_simulation()
#    
#    # Display summary
#    print("\n" + "="*60)
#    print("SIMULATION SUMMARY")
#    print("="*60)
#    print(f"Total simulations: {len(df)}")
#    print(f"Methods: {sorted(df['method'].unique())}")
#    print(f"Sample sizes: {sorted(df['n'].unique())}")
#    print(f"Features (p): {sorted(df['p'].unique())}")
#    print(f"MSE range: {df['mse'].min():.4f} - {df['mse'].max():.4f}")
#    print("="*60)
#    
#    # Optionally run evaluation
#    run_eval = input("\nRun evaluation plots? (y/n): ").strip().lower()
#    if run_eval in ['y', 'yes']:
#        from evaluation import create_all_plots
#        csv_path = 'artifacts/full_simulation_results.csv'
#        create_all_plots(csv_path)
