import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read data from Excel
# Make sure to have the 'distillation_data.xlsx' file in the same directory
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Year 3 distilattion optimisation/distillation_data.csv')

# Step 2: Set a fixed experiment time (in seconds) for volume calculations
fixed_time = 3600  # For example, 1 hour (3600 seconds)

# Step 3: Calculate volumes from flow rates
# Assuming flow rates are given in liters per hour (L/h)
data['Reflux Volume'] = data['Reflux Flow Rate'] * (fixed_time / 3600)
data['Distillate Volume'] = data['Distillate Flow Rate'] * (fixed_time / 3600)
# The volumes are now in liters, calculated over the fixed experiment time

# Step 4: Calculate reflux ratio from the flow rates
data['Reflux Ratio'] = data['Reflux Flow Rate'] / data['Distillate Flow Rate']
# Reflux Ratio is a key parameter in distillation, influencing purity and energy consumption

# Step 5: Prepare inputs for interpolation functions
reflux_ratios = data['Reflux Ratio'].values  # Array of reflux ratios
temperatures = data['Temperature'].values    # Array of temperatures in degrees Celsius

# Purity and energy consumption are collected as they are
purities = data['Purity'].values
# Energy consumption is calculated from real-world data: sum of boiler and condenser duties in watts
# Assuming 'Boiler Duty' and 'Condenser Duty' are columns in your data
energy_consumptions = data['Boiler Duty'].values + data['Condenser Duty'].values

# Step 6: Combine reflux ratio and temperature into a 2D input space for interpolation
inputs = np.column_stack((reflux_ratios, temperatures))
# This creates a 2D array where each row is [reflux_ratio, temperature]

# Step 7: Create interpolation functions for purity and energy consumption
# These functions estimate purity and energy consumption for any given reflux ratio and temperature
purity_function = LinearNDInterpolator(inputs, purities)
energy_function = LinearNDInterpolator(inputs, energy_consumptions)
# LinearNDInterpolator performs linear interpolation over a N-dimensional space

# Step 8: Define the penalty function for low purity
def penalty_for_low_purity(purity):
    desired_purity = 0.95  # Target purity level (95%)
    if purity < desired_purity:
        # Apply a large penalty proportional to the shortfall in purity
        return 1e6 * (desired_purity - purity)
    else:
        # No penalty if the desired purity is met or exceeded
        return 0

# Step 9: Define the cost function WITHOUT the decorator
def cost_function(reflux_flow_rate, distillate_flow_rate, temperature):
    # Calculate the reflux ratio for the given flow rates
    reflux_ratio = reflux_flow_rate / distillate_flow_rate

    # Predict purity and energy consumption using the interpolation functions
    purity = purity_function(reflux_ratio, temperature)
    energy = energy_function(reflux_ratio, temperature)

    # Handle cases where the interpolation returns NaN or None
    if purity is None or np.isnan(purity) or energy is None or np.isnan(energy):
        return 1e8  # Assign a large cost to invalid or out-of-bounds inputs

    # Calculate total cost as energy consumption plus any penalty for low purity
    total_cost = energy + penalty_for_low_purity(purity)
    return total_cost
total_costs = energy_consumptions.copy()
for i, purity in enumerate(purities):
    penalty = penalty_for_low_purity(purity)
    total_costs[i] += penalty

# Create a wrapper function with the decorator for use with gp_minimize
@use_named_args([
    Real(data['Reflux Flow Rate'].min(), data['Reflux Flow Rate'].max(), name='reflux_flow_rate'),
    Real(data['Distillate Flow Rate'].min(), data['Distillate Flow Rate'].max(), name='distillate_flow_rate'),
    Real(data['Temperature'].min(), data['Temperature'].max(), name='temperature')
])
def objective_function(**params):
    return cost_function(
        params['reflux_flow_rate'],
        params['distillate_flow_rate'],
        params['temperature']
    )

# Step 10: Define the search space for optimization based on data ranges
search_space = [
    Real(data['Reflux Flow Rate'].min(), data['Reflux Flow Rate'].max(), name='reflux_flow_rate'),
    Real(data['Distillate Flow Rate'].min(), data['Distillate Flow Rate'].max(), name='distillate_flow_rate'),
    Real(data['Temperature'].min(), data['Temperature'].max(), name='temperature')
]

# Step 11: Run Bayesian Optimization using the objective_function
res = gp_minimize(
    func=objective_function,    # Use the wrapper function
    dimensions=search_space,
    n_calls=50,
    random_state=42
)


# Step 12: Extract and display the optimal parameters and minimum cost
optimal_reflux_flow_rate = res.x[0]
optimal_distillate_flow_rate = res.x[1]
optimal_temperature = res.x[2]
minimum_cost = res.fun

print(f"Optimal reflux flow rate: {optimal_reflux_flow_rate:.4f} L/h")
print(f"Optimal distillate flow rate: {optimal_distillate_flow_rate:.4f} L/h")
print(f"Optimal temperature: {optimal_temperature:.2f} °C")
print(f"Minimum cost (Energy + Penalty): {minimum_cost:.2f} Watts")

# Step 13: Visualization of Cost vs Iterations
plt.figure(figsize=(10, 6))
plt.plot(res.func_vals, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total Cost (Watts)')
plt.title('Convergence of Bayesian Optimization')
plt.grid(True)
plt.show()
# This plot shows how the cost decreases over each iteration, indicating the optimizer's progress

# Step 14: Visualization of Cost Function over Reflux Flow Rates
# Generate a range of reflux flow rates for plotting
reflux_flow_rates_plot = np.linspace(
    data['Reflux Flow Rate'].min(),
    data['Reflux Flow Rate'].max(),
    100
)
constant_distillate_flow_rate = optimal_distillate_flow_rate
constant_temperature = optimal_temperature

# Calculate the cost for each reflux flow rate in the range
costs = []
for r in reflux_flow_rates_plot:
    params = {
        'reflux_flow_rate': r,
        'distillate_flow_rate': constant_distillate_flow_rate,
        'temperature': constant_temperature
    }
    cost = cost_function(**params)  # Now works because cost_function accepts keyword arguments
    costs.append(cost)

# Plot the cost function against reflux flow rate
plt.figure(figsize=(10, 6))
plt.plot(reflux_flow_rates_plot, costs, label='Cost Function')
plt.axvline(x=optimal_reflux_flow_rate, color='red', linestyle='--', label='Optimal Reflux Flow Rate')
plt.xlabel('Reflux Flow Rate (L/h)')
plt.ylabel('Total Cost (Watts)')
plt.title(f'Cost Function vs Reflux Flow Rate at Distillate Flow Rate = {constant_distillate_flow_rate:.2f} L/h and Temperature = {constant_temperature:.2f} °C')
plt.legend()
plt.grid(True)
plt.show()
# This plot helps visualize how the cost varies with reflux flow rate at the optimal distillate flow rate and temperature

# Step 14: Visualization of Cost Function over Reflux Flow Rates
# Generate a range of reflux flow rates for plotting
reflux_flow_rates_plot = np.linspace(
    data['Reflux Flow Rate'].min(),
    data['Reflux Flow Rate'].max(),
    100
)
# Use the optimal distillate flow rate and temperature for the plot
constant_distillate_flow_rate = optimal_distillate_flow_rate
constant_temperature = optimal_temperature

# Calculate the cost for each reflux flow rate in the range
costs = []
for r in reflux_flow_rates_plot:
    params = {
        'reflux_flow_rate': r,
        'distillate_flow_rate': constant_distillate_flow_rate,
        'temperature': constant_temperature
    }
    cost = cost_function(**params)
    costs.append(cost)

# Plot the cost function against reflux flow rate
plt.figure(figsize=(10, 6))
plt.plot(reflux_flow_rates_plot, costs, label='Cost Function')
plt.axvline(x=optimal_reflux_flow_rate, color='red', linestyle='--', label='Optimal Reflux Flow Rate')
plt.xlabel('Reflux Flow Rate (L/h)')
plt.ylabel('Total Cost (Watts)')
plt.title(f'Cost Function vs Reflux Flow Rate at Distillate Flow Rate = {constant_distillate_flow_rate:.2f} L/h and Temperature = {constant_temperature:.2f} °C')
plt.legend()
plt.grid(True)
plt.show()
# This plot helps visualize how the cost varies with reflux flow rate at the optimal distillate flow rate and temperature

# Step 15: New Plot - Visualizing the Optimal Point with Respect to the Data
# Create a 3D scatter plot of the experimental data, coloring points by total cost
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the experimental data points
scatter = ax.scatter(
    data['Reflux Ratio'],
    data['Temperature'],
    total_costs,
    c=total_costs,
    cmap='viridis',
    alpha=0.7,
    edgecolor='k',
    label='Experimental Data'
)

# Add a color bar to indicate the cost
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Total Cost (Watts)', rotation=270, labelpad=15)

# Plot the optimal point
optimal_reflux_ratio = optimal_reflux_flow_rate / optimal_distillate_flow_rate
optimal_cost = minimum_cost

ax.scatter(
    optimal_reflux_ratio,
    optimal_temperature,
    optimal_cost,
    color='red',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

# Set labels and title
ax.set_xlabel('Reflux Ratio')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Total Cost (Watts)')
ax.set_title('Optimal Point in Relation to Experimental Data')

# Add legend
ax.legend()

plt.show()
# This 3D plot shows the experimental data points and highlights the optimal point found by the optimization
