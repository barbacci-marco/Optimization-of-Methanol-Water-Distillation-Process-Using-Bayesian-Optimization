import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Import interpolation function for multi-dimensional data
from scipy.interpolate import LinearNDInterpolator

# Import 3D plotting toolkit for matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read Data from CSV File
# -------------------------------
# Load the experimental distillation data from a CSV file into a pandas DataFrame.
# Ensure that the CSV file is located at the specified path.
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Lab_distillation_data.csv')

# Step 2: Extract Data Columns
# ----------------------------
# Extract relevant columns from the DataFrame and convert them to NumPy arrays for processing.

# Methanol purity percentages (%)
purities = data['Methanol Purity (%)'].values

# Distillate flow rates (ml/h)
Distillate_flowrate = data['Distillate Flowrate (ml/h)'].values

# Reflux flow rates (ml/h)
Reflux_flowrate = data['Reflux Flowrate (ml/h)'].values

# Time taken to reach 1.6 liters of distillate (hours)
Time = data['Time to Reach 1.6 L (hours)'].values

# Pump power outage per 1.6 liters of distillate (kW/h)
Pump_power_Outage = data['Pump Power Outage for 1.6L Distilate'].values

# Condenser duty (kW/h)
Condenser_Duty = data['Water Cooling Heat Transfer (kW/h)'].values

# Reboiler duty (kW/h)
Reboiler_Duty = data['Electric Reboiler Duty (kW/h)'].values

# Step 3: Calculate Total Energy Consumption
# ------------------------------------------
# Sum the energy consumptions from the reboiler, condenser, and pump to get the total energy consumption.

# Total energy consumption (kW/h)
energy_consumptions = Reboiler_Duty + Condenser_Duty + Pump_power_Outage

# Step 4: Prepare Input Data for Interpolation
# --------------------------------------------
# Combine the distillate and reflux flow rates into a 2D array of input pairs for interpolation.

# Create input pairs: each row is [Distillate_flowrate, Reflux_flowrate]
inputs = np.column_stack((Distillate_flowrate, Reflux_flowrate))

# Ensure that the number of input pairs matches the number of purity measurements.
if inputs.shape[0] != purities.shape[0]:
    raise ValueError("Mismatch between number of input points and purity measurements.")

# Step 5: Create Interpolation Functions
# --------------------------------------
# Use linear interpolation to create functions that predict purity and energy consumption
# for any given distillate and reflux flow rates within the data range.

# Interpolation function for methanol purity (%)
purity_function = LinearNDInterpolator(inputs, purities)

# Interpolation function for total energy consumption (kW/h)
energy_function = LinearNDInterpolator(inputs, energy_consumptions)

# Step 6: Define the Penalty Function for Low Purity
# --------------------------------------------------
# Define a function that applies a penalty if the purity is below the desired level.

def penalty_for_low_purity(purity):
    desired_purity = 95  # Target purity level (%)
    if purity < desired_purity:
        # Apply a large penalty proportional to the shortfall in purity
        return 1e6 * (desired_purity - purity)
    else:
        # No penalty if the desired purity is met or exceeded
        return 0

# Step 7: Compute Reflux Ratio from Existing Data
# -----------------------------------------------
# Calculate the reflux ratio for each data point.

# Reflux ratio (dimensionless)
reflux_ratio = Reflux_flowrate / Distillate_flowrate

# Step 8: Calculate Total Costs for Existing Data
# -----------------------------------------------
# For each data point, calculate the total cost as the sum of energy consumption
# and any penalty for not achieving the desired purity.

# Initialize total costs with energy consumptions
total_costs = energy_consumptions.copy()

# Loop over each data point to apply the penalty function
for i, purity in enumerate(purities):
    penalty = penalty_for_low_purity(purity)
    total_costs[i] += penalty  # Add penalty to the energy consumption

# Step 9: Define the Cost Function
# --------------------------------
# Define the objective function that will be minimized during optimization.

def cost_function(Reflux_flowrate, Distillate_flowrate):
    # Calculate the reflux ratio for the given flow rates
    reflux_ratio = Reflux_flowrate / Distillate_flowrate

    # Predict purity and energy consumption using the interpolation functions
    purity = purity_function(Distillate_flowrate, Reflux_flowrate)
    energy = energy_function(Distillate_flowrate, Reflux_flowrate)

    # Handle cases where the interpolation returns NaN or None (out-of-bounds inputs)
    if purity is None or np.isnan(purity) or energy is None or np.isnan(energy):
        return 1e8  # Assign a large cost to invalid inputs

    # Calculate total cost as energy consumption plus any penalty for low purity
    total_cost = energy + penalty_for_low_purity(purity)
    return total_cost

# Step 10: Create a Wrapper Function with the Decorator for Optimization
# ----------------------------------------------------------------------
# Use the 'use_named_args' decorator to map the optimization parameters to function arguments.

@use_named_args([
    Real(Reflux_flowrate.min(), Reflux_flowrate.max(), name='Reflux_flowrate'),           # Reflux flow rate range
    Real(Distillate_flowrate.min(), Distillate_flowrate.max(), name='Distillate_flowrate') # Distillate flow rate range
])
def objective_function(**params):
    # The objective function to minimize; it calls the cost_function with parameters provided by the optimizer.
    return cost_function(
        params['Reflux_flowrate'],
        params['Distillate_flowrate'],
    )

# Step 11: Define the Search Space for Optimization
# -------------------------------------------------
# Specify the bounds for each parameter based on the observed data ranges.

search_space = [
    Real(Reflux_flowrate.min(), Reflux_flowrate.max(), name='Reflux_flowrate'),
    Real(Distillate_flowrate.min(), Distillate_flowrate.max(), name='Distillate_flowrate'),
]

# Step 12: Run Bayesian Optimization
# ----------------------------------
# Use Gaussian Process-based optimization to find the parameters that minimize the cost function.

res = gp_minimize(
    func=objective_function,    # The function to minimize
    dimensions=search_space,    # The parameter space
    n_calls=100,                # Number of function evaluations
    random_state=42             # Seed for reproducibility
)

# Step 13: Extract and Display the Optimal Parameters and Minimum Cost
# --------------------------------------------------------------------
# Retrieve the optimal reflux and distillate flow rates and the minimum cost achieved.

optimal_reflux_flow_rate = res.x[0]        # Optimal reflux flow rate (ml/h)
optimal_distillate_flow_rate = res.x[1]    # Optimal distillate flow rate (ml/h)
minimum_cost = res.fun                     # Minimum total cost (kW/h)

# Print the optimization results
print(f"Optimal reflux flow rate: {optimal_reflux_flow_rate:.4f} ml/h")
print(f"Optimal distillate flow rate: {optimal_distillate_flow_rate:.4f} ml/h")
print(f"Minimum cost (Energy + Penalty): {minimum_cost:.2f} kW/h")

# Step 14: Visualization of Cost vs. Iterations
# ---------------------------------------------
# Plot the cost function values at each iteration to observe the convergence of the optimizer.

plt.figure(figsize=(10, 6))
plt.plot(res.func_vals, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total Cost (kW/h)')
plt.title('Convergence of Bayesian Optimization')
plt.yscale('log')  # Use logarithmic scale due to potentially large cost variations
plt.grid(True)
plt.show()

# Step 15: Visualization of Cost Function over Reflux Flow Rates
# --------------------------------------------------------------
# Analyze how the cost varies with reflux flow rate at the optimal distillate flow rate.

# Generate a range of reflux flow rates for plotting
reflux_flow_rates_plot = np.linspace(
    Reflux_flowrate.min(),
    Reflux_flowrate.max(),
    200  # Number of points in the range
)

# Use the optimal distillate flow rate for the analysis
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
plt.title(f'Cost Function vs. Reflux Flow Rate at Distillate Flow Rate = {constant_distillate_flow_rate:.2f} ml/h')
plt.legend()
plt.grid(True)
plt.show()

# Step 16: Visualization of the Optimal Point with Respect to the Data
# --------------------------------------------------------------------
# Create a 3D scatter plot to visualize the experimental data and the optimal point.

# Initialize the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the experimental data points, coloring them by methanol purity (%)
scatter = ax.scatter(
    reflux_ratio,          # X-axis: Reflux Ratio (dimensionless)
    Distillate_flowrate,   # Y-axis: Distillate Flow Rate (ml/h)
    total_costs,           # Z-axis: Total Cost (kW/h)
    c=purities,            # Color mapping based on methanol purity (%)
    cmap='viridis',        # Colormap for the scatter points
    alpha=0.7,
    edgecolor='k',
    label='Experimental Data'
)

# Add a color bar to indicate the methanol purity values
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Methanol Purity (%)', rotation=270, labelpad=15)

# Calculate the optimal reflux ratio
optimal_reflux_ratio = optimal_reflux_flow_rate / optimal_distillate_flow_rate

# Plot the optimal point found by the optimization
ax.scatter(
    optimal_reflux_ratio,          # X-coordinate: Optimal Reflux Ratio
    optimal_distillate_flow_rate,  # Y-coordinate: Optimal Distillate Flow Rate (ml/h)
    minimum_cost,                  # Z-coordinate: Minimum Total Cost (kW/h)
    color='red',
    s=100,                         # Size of the point
    label='Optimal Point',
    edgecolor='k'
)

# Set axis labels and title
ax.set_xlabel('Reflux Ratio')
ax.set_ylabel('Distillate Flow Rate (ml/h)')
ax.set_zlabel('Total Cost (kW/h)')
ax.set_title('Optimal Point in Relation to Experimental Data')
ax.legend() # Add legend to distinguish experimental data and the optimal point
plt.show()
