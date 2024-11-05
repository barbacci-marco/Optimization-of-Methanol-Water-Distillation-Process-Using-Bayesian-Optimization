import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read data from CSV
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Lab_distillation_data.csv')

# Extract data columns
purities = data['Methanol Purity (%)'].values
Distillate_flowrate = data['Distillate Flowrate (ml/h)'].values
Reflux_flowrate = data['Reflux Flowrate (ml/h)'].values
Time = data['Time to Reach 1.6 L (hours)'].values
Pump_power_Outage = data['Pump Power Outage for 1.6L Distilate'].values
Condenser_Duty = data['Water Cooling Heat Transfer (kW/h)'].values
Reboiler_Duty = data['Electric Reboiler Duty (kW/h)'].values

# Calculate total energy consumption
energy_consumptions = Reboiler_Duty + Condenser_Duty + Pump_power_Outage

# Prepare input data for interpolation
inputs = np.column_stack((Distillate_flowrate, Reflux_flowrate))

# Ensure that inputs and outputs are of compatible shapes
if inputs.shape[0] != purities.shape[0]:
    raise ValueError("Mismatch between number of input points and purity measurements.")

# Create interpolation functions for purity and energy consumption
purity_function = LinearNDInterpolator(inputs, purities)
energy_function = LinearNDInterpolator(inputs, energy_consumptions)

# Define the penalty function for low purity
def penalty_for_low_purity(purity):
    desired_purity = 95  # Target purity level (%)
    if purity < desired_purity:
        return 1e6 * (desired_purity - purity)
    else:
        return 0

# Compute reflux ratio from existing data
reflux_ratio = Reflux_flowrate / Distillate_flowrate

# Calculate total costs for existing data
total_costs = energy_consumptions.copy()
for i, purity in enumerate(purities):
    penalty = penalty_for_low_purity(purity)
    total_costs[i] += penalty

# Define the cost function
def cost_function(Reflux_flowrate, Distillate_flowrate):
    # Calculate the reflux ratio
    reflux_ratio = Reflux_flowrate / Distillate_flowrate

    # Predict purity and energy consumption using the interpolation functions
    purity = purity_function(Distillate_flowrate, Reflux_flowrate)
    energy = energy_function(Distillate_flowrate, Reflux_flowrate)

    # Handle cases where the interpolation returns NaN or None
    if purity is None or np.isnan(purity) or energy is None or np.isnan(energy):
        return 1e8  # Assign a large cost to invalid or out-of-bounds inputs

    # Calculate total cost as energy consumption plus any penalty for low purity
    total_cost = energy + penalty_for_low_purity(purity)
    return total_cost

# Create a wrapper function with the decorator for use with gp_minimize
@use_named_args([
    Real(Reflux_flowrate.min(), Reflux_flowrate.max(), name='Reflux_flowrate'),
    Real(Distillate_flowrate.min(), Distillate_flowrate.max(), name='Distillate_flowrate')
])
def objective_function(**params):
    return cost_function(
        params['Reflux_flowrate'],
        params['Distillate_flowrate'],
    )

# Define the search space for optimization based on data ranges
search_space = [
    Real(Reflux_flowrate.min(), Reflux_flowrate.max(), name='Reflux_flowrate'),
    Real(Distillate_flowrate.min(), Distillate_flowrate.max(), name='Distillate_flowrate'),
]

# Run Bayesian Optimization using the objective_function
res = gp_minimize(
    func=objective_function,    # Use the wrapper function
    dimensions=search_space,
    n_calls=100,
    random_state=42
)

# Extract and display the optimal parameters and minimum cost
optimal_reflux_flow_rate = res.x[0]
optimal_distillate_flow_rate = res.x[1]
minimum_cost = res.fun

print(f"Optimal reflux flow rate: {optimal_reflux_flow_rate:.4f} ml/h")
print(f"Optimal distillate flow rate: {optimal_distillate_flow_rate:.4f} ml/h")
print(f"Minimum cost (Energy + Penalty): {minimum_cost:.2f} kW/h")

# Visualization of Cost vs Iterations
plt.figure(figsize=(10, 6))
plt.plot(res.func_vals, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total Cost (kW/h)')
plt.title('Convergence of Bayesian Optimization')
plt.yscale('log')
plt.grid(True)
plt.show()

# Visualization of Cost Function over Reflux Flow Rates
# Generate a range of reflux flow rates for plotting
reflux_flow_rates_plot = np.linspace(
    Reflux_flowrate.min(),
    Reflux_flowrate.max(),
    200
)
# Use the optimal distillate flow rate for the plot
constant_distillate_flow_rate = optimal_distillate_flow_rate

# Calculate the cost for each reflux flow rate in the range
costs = []
for r in reflux_flow_rates_plot:
    cost = cost_function(r, constant_distillate_flow_rate)
    costs.append(cost)

# Plot the cost function against reflux flow rate
plt.figure(figsize=(10, 6))
plt.plot(reflux_flow_rates_plot, costs, label='Cost Function')
plt.axvline(x=optimal_reflux_flow_rate, color='red', linestyle='--', label='Optimal Reflux Flow Rate')
plt.xlabel('Reflux Flow Rate (ml/h)')
plt.ylabel('Total Cost (kW/h)')
plt.title(f'Cost Function vs Reflux Flow Rate at Distillate Flow Rate = {constant_distillate_flow_rate:.2f} ml/h')
plt.legend()
plt.grid(True)
plt.show()

# Visualization of the Optimal Point with Respect to the Data
# Create a 3D scatter plot of the experimental data, coloring points by total cost
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the experimental data points
scatter = ax.scatter(
    reflux_ratio,
    Distillate_flowrate,
    total_costs,
    c= purities,
    cmap='viridis',
    alpha=0.7,
    edgecolor='k',
    label='Experimental Data'
)

# Add a color bar to indicate the cost
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Methanol Purity (%)', rotation=270, labelpad=15)

# Plot the optimal point
optimal_reflux_ratio = optimal_reflux_flow_rate / optimal_distillate_flow_rate
optimal_cost = minimum_cost

ax.scatter(
    optimal_reflux_ratio,
    optimal_distillate_flow_rate,
    optimal_cost,
    color='red',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

# Set labels and title
ax.set_xlabel('Reflux Ratio')
ax.set_ylabel('Distillate Flow Rate (ml/h)')
ax.set_zlabel('Total Cost (kW/h)')
ax.set_title('Optimal Point in Relation to Experimental Data')

# Add legend
ax.legend()

plt.show()
