# Stats 607 Studios 7 - Regression Methods Simulation

# Group Members

Chandler Nielsen <br>
Abhiti Mishra <br>
Jingyuan Yang <br>
Dili Maduabum <br>

## Quick Start

Navigate to project root and run:
```bash
cd Stats-607-Studios-7
python src/run_simulation.py
```

This will:
1. Generate simulated data, conduct OLS, LAD, and Huber regression, and compare the results.
2. Save results to `artifacts/full_simulation_results.csv`
3. Create visualizations in `figures/` folder

## File Structure

- `src/` - All source code
  - `run_simulation.py` - Main simulation runner
  - `generate_data.py` - Data generation with various noise distributions
  - `run_reg.py` - Regression method implementations
  - `evaluation.py` - MSE calculation and plotting functions

- `artifacts/` - Simulation results (CSV files with MSE data)
- `figures/` - Generated plots and visualizations

## Data Format

Simulation results CSV contains: `method`, `n`, `p`, `df`, `rho`, `SNR`, `rep`, `mse`
