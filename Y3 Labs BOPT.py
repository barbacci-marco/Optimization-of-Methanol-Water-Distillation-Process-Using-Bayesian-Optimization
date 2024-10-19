import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.interpolate import LinearNDInterpolator

# Step 1: Read data from Excel
# Make sure to have the 'distillation_data.xlsx' file in the same directory
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Year 3 distilattion optimisation/distillation_data.csv')

# Step 2: Calculate reflux ratio from the flow rates
data['Reflux Ratio'] = data['Reflux Flow Rate'] / data['Distillate Flow Rate']

# Step 3: Define interpolation functions using reflux ratio and temperature
reflux_ratios = data['Reflux Ratio'].values
temperatures = data['Temperature'].values
purities = data['Purity'].values
energy_consumptions = data['Energy Consumption'].values

# Combine reflux ratio and temperature into a 2D input space
inputs = np.column_stack((reflux_ratios, temperatures))

# Create interpolation functions for purity and energy consumption
purity_function = LinearNDInterpolator(inputs, purities)
energy_function = LinearNDInterpolator(inputs, energy_consumptions)

# Step 4: Define the cost function for optimization
def penalty_for_low_purity(purity):
    # Simulate a penalty if purity is below a certain threshold
    if purity < 0.95:
        return 1000 * (0.95 - purity)  # Heavy penalty for low purity
    else:
        return 0  # No penalty if purity is acceptable

@use_named_args([Real(0.5, 3.0, name='reflux_flow_rate'), Real(1.0, 3.0, name='distillate_flow_rate'), Real(94.0, 96.0, name='temperature')])
def cost_function(reflux_flow_rate, distillate_flow_rate, temperature):
    # Calculate the reflux ratio based on the flow rates
    reflux_ratio = reflux_flow_rate / distillate_flow_rate
    
    # Use the reflux ratio and temperature to get purity and energy consumption
    purity = purity_function(reflux_ratio, temperature)  # Get purity from the interpolation function
    energy = energy_function(reflux_ratio, temperature)  # Get energy consumption from the interpolation function
      # Handle NaN values
    # If NaN or invalid values are encountered, return a large cost instead of infinity
    if pd.isna(purity) or pd.isna(energy):
        return 1e6  # Large penalty for invalid points
    total_cost = energy + penalty_for_low_purity(purity)
    return total_cost

# Step 5: Run Bayesian Optimization with 3 variables: reflux flow rate, distillate flow rate, and temperature
res = gp_minimize(cost_function, [Real(0.5, 3.0), Real(1.0, 3.0), Real(94.0, 96.0)], n_calls=20, random_state=0)

# Output the optimal flow rates, temperature, and corresponding minimum cost
print(f"Optimal reflux flow rate: {res.x[0]}")
print(f"Optimal distillate flow rate: {res.x[1]}")
print(f"Optimal temperature: {res.x[2]}")
print(f"Minimum cost: {res.fun}")

# Step 6: Visualization
# Plot cost as a function of iteration
plt.figure(figsize=(8, 6))
plt.plot(res.func_vals)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iterations in Bayesian Optimization')
plt.grid(True)
plt.show()

# Plot the cost function over a range of reflux flow rates at a constant distillate flow rate and temperature (for visualization)
reflux_flow_rates_plot = np.linspace(0.5, 3.0, 100)
constant_distillate_flow_rate = 2.0  # You can change this to a specific value to visualize at a given distillate flow rate
constant_temperature = 95.0  # You can change this to a specific value to visualize at a given temperature
costs = [cost_function(r, constant_distillate_flow_rate, constant_temperature) for r in reflux_flow_rates_plot]

plt.figure(figsize=(8, 6))
plt.plot(reflux_flow_rates_plot, costs, label=f'Cost Function at Distillate Flow = {constant_distillate_flow_rate} and Temp = {constant_temperature}°C')
plt.axvline(x=res.x[0], color='red', linestyle='--', label=f'Optimal Reflux Flow Rate: {res.x[0]:.2f}')
plt.xlabel('Reflux Flow Rate')
plt.ylabel('Cost')
plt.title('Cost Function vs Reflux Flow Rate at Constant Distillate Flow Rate and Temperature')
plt.legend()
plt.grid(True)
plt.show()
