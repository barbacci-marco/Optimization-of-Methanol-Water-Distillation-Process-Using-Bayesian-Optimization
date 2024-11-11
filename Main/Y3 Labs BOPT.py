import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize                       # Import functions for Bayesian optimization
from skopt.space import Real
from skopt.utils import use_named_args
from mpl_toolkits.mplot3d import Axes3D             # Import 3D plotting toolkit for matplotlib

# Step 1: Read Data from CSV File
# -------------------------------
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Lab_distillation_data.csv')

# Step 2: Extract Data Columns
# ----------------------------
# Methanol purity percentages (%)
purities_percent = data['Methanol Purity (%)'].values
purities = purities_percent / 100  # Convert to fraction

# Distillate and reflux flow rates in liters per hour (L/h)
Distillate_flowrate = data['Distillate Flowrate (l/h)'].values
Reflux_flowrate = data['Reflux Flowrate (l/h)'].values

# Time taken to reach 1.6 liters of distillate (hours)
Time = data['Time to Reach 1.6 L (hours)'].values

# Energy consumption components in kWh per batch
Pump_power_Outage = data['Pump Power Outage for 1.6L Distillate (kW/h)'].values
Condenser_Duty = data['Water Cooling Heat Transfer (kW/h)'].values
Reboiler_Duty = data['Electric Reboiler Duty (kW/h)'].values

# Step 3: Calculate Total Energy Consumption per Batch
# ----------------------------------------------------
# Total energy consumption per batch (kWh per batch)
energy_consumptions = Reboiler_Duty + Condenser_Duty + Pump_power_Outage

# Step 4: Prepare Input Data for Polynomial Regression
# ----------------------------------------------------
inputs = np.column_stack((Distillate_flowrate, Reflux_flowrate))

# Ensure that the number of input pairs matches the number of purity measurements.
if inputs.shape[0] != purities.shape[0]:
    raise ValueError("Mismatch between number of input points and purity measurements.")

# Check for NaNs in inputs and outputs
valid_indices = (
    ~np.isnan(inputs).any(axis=1) &
    ~np.isnan(purities) &
    ~np.isnan(energy_consumptions)
)

# Filter out invalid data
inputs = inputs[valid_indices]
purities = purities[valid_indices]
purities_percent = purities_percent[valid_indices]
energy_consumptions = energy_consumptions[valid_indices]
Distillate_flowrate = Distillate_flowrate[valid_indices]
Reflux_flowrate = Reflux_flowrate[valid_indices]
Time = Time[valid_indices]

# Step 5: Create Polynomial Features and Fit Polynomial Regression Models
# -----------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features up to degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
inputs_poly = poly.fit_transform(inputs)

# Fit the polynomial regression model for purity
purity_model = LinearRegression()
purity_model.fit(inputs_poly, purities)

# Fit the polynomial regression model for energy consumption
energy_model = LinearRegression()
energy_model.fit(inputs_poly, energy_consumptions)

# Step 6: Define the Purity Function and Energy Function Using the Fitted Models
# ------------------------------------------------------------------------------

def purity_function(Distillate_flowrate, Reflux_flowrate):
    # Ensure inputs are numpy arrays
    Distillate_flowrate = np.asarray(Distillate_flowrate)
    Reflux_flowrate = np.asarray(Reflux_flowrate)

    # Flatten the input arrays to 1D
    D_flat = Distillate_flowrate.ravel()
    R_flat = Reflux_flowrate.ravel()

    # Stack the flattened inputs column-wise
    input_data = np.column_stack((D_flat, R_flat))

    # Transform the input data
    input_data_poly = poly.transform(input_data)

    # Predict the purity
    purity_flat = purity_model.predict(input_data_poly)

    # Reshape the output to match the input shape
    purity = purity_flat.reshape(Distillate_flowrate.shape)

    # Clip purity values to [0, 1]
    purity = np.clip(purity, 0, 1)

    return purity

def energy_function(Distillate_flowrate, Reflux_flowrate):
    # Ensure inputs are numpy arrays
    Distillate_flowrate = np.asarray(Distillate_flowrate)
    Reflux_flowrate = np.asarray(Reflux_flowrate)

    # Flatten the input arrays to 1D
    D_flat = Distillate_flowrate.ravel()
    R_flat = Reflux_flowrate.ravel()

    # Stack the flattened inputs column-wise
    input_data = np.column_stack((D_flat, R_flat))

    # Transform the input data
    input_data_poly = poly.transform(input_data)

    # Predict the energy consumption
    energy_flat = energy_model.predict(input_data_poly)

    # Reshape the output to match the input shape
    energy = energy_flat.reshape(Distillate_flowrate.shape)

    # Ensure energy consumption is non-negative
    energy = np.maximum(energy, 0)

    return energy

# Step 7: Define the Pricing Function Based on Purity
# ---------------------------------------------------
def price_of_methanol(purity):
    if purity < 0.85:
        return 0  # Price is £0 for purity below 85%
    else:
        # Exponential pricing function
        k = 25  # Adjusted k for better curve fitting
        numerator = 1 - np.exp(-k * (purity - 0.85))
        denominator = 1 - np.exp(-k * (0.9985 - 0.85))
        price = 23 *26 * (numerator / denominator)
        return price  # Price in £ per liter

# Step 8: Compute Reflux Ratio from Existing Data
# -----------------------------------------------
reflux_ratio = Reflux_flowrate / Distillate_flowrate

# Step 9: Calculate Objective Values for Existing Data
# ----------------------------------------------------
objective_values = []
electricity_cost_per_kWh = 0.04  # £ per kWh

for i in range(len(purities)):
    purity = purities[i]
    energy_per_batch = energy_consumptions[i]  # kWh per batch
    distillate_flow_rate = Distillate_flowrate[i]  # L/h

    # Calculate the price of methanol based on purity
    price = price_of_methanol(purity)  # £ per liter

    # Revenue per batch
    revenue = price * 1.6  # £ per batch

    # Energy cost per batch
    energy_cost = energy_per_batch * electricity_cost_per_kWh  # £ per batch

    # Profit per batch
    profit = revenue - energy_cost

    # Append the negative profit to objective_values
    objective_values.append(-profit)  # Negative for consistency with plotting

objective_values = np.array(objective_values)

# Step 10: Define the Cost Function
# ---------------------------------
def cost_function(params):
    reflux_ratio = params[0]
    distillate_flow_rate = params[1]

    # Compute Reflux_flowrate based on reflux_ratio and distillate_flow_rate
    reflux_flow_rate = reflux_ratio * distillate_flow_rate

    # Predict purity and energy consumption
    purity = purity_function(distillate_flow_rate, reflux_flow_rate)
    energy_per_batch = energy_function(distillate_flow_rate, reflux_flow_rate)

    # Handle cases where the predictions are invalid
    if np.isnan(purity).any() or np.isnan(energy_per_batch).any():
        return 1e8  # Assign a large cost to invalid inputs

    # Energy cost per batch
    energy_cost = energy_per_batch * electricity_cost_per_kWh  # £ per batch

    # Revenue per batch
    price = price_of_methanol(purity)  # £ per liter
    revenue = price * 1.6  # £ per batch

    # Total profit per batch
    profit = revenue - energy_cost

    # If profit is an array (due to array inputs), take the mean
    if np.size(profit) > 1:
        profit = np.mean(profit)

    # Return negative profit to convert to a minimization problem
    objective_value = -profit

    return objective_value

# Step 11: Define the Search Space for Optimization
# -------------------------------------------------
# Define bounds for reflux ratio and distillate flow rate based on your data
reflux_ratio_min = reflux_ratio.min()
reflux_ratio_max = reflux_ratio.max()
distillate_flow_rate_min = Distillate_flowrate.min()
distillate_flow_rate_max = Distillate_flowrate.max()

search_space = [
    Real(reflux_ratio_min, reflux_ratio_max, name='reflux_ratio'),
    Real(distillate_flow_rate_min, distillate_flow_rate_max, name='distillate_flow_rate')
]

# Step 12: Create a Wrapper Function with the Decorator for Optimization
# ----------------------------------------------------------------------
@use_named_args(search_space)
def objective_function(**params):
    return cost_function([params['reflux_ratio'], params['distillate_flow_rate']])

# Step 13: Run Bayesian Optimization
# ----------------------------------
res = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=50,
    random_state=42,
    base_estimator=None,
    n_random_starts=None,
    n_initial_points=10,
    initial_point_generator="random",
    acq_func="gp_hedge", 
    acq_optimizer="auto", 
    x0=None, y0=None,
    verbose=False, callback=None,
    n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96,
    noise="gaussian", n_jobs=1, model_queue_size=None
)

# Step 14: Extract and Display the Optimal Parameters and Maximum Profit
# ----------------------------------------------------------------------
optimal_reflux_ratio = res.x[0]
optimal_distillate_flow_rate = res.x[1]
optimal_reflux_flow_rate = optimal_reflux_ratio * optimal_distillate_flow_rate
maximum_profit = -res.fun  # Maximum total profit (£ per batch)

print(f"Optimal reflux ratio: {optimal_reflux_ratio:.4f}")
print(f"Optimal distillate flow rate: {optimal_distillate_flow_rate:.4f} L/h")
print(f"Optimal reflux flow rate: {optimal_reflux_flow_rate:.4f} L/h")
print(f"Maximum profit per batch: £{maximum_profit:.2f}")

# Step 15: Visualization of Profit vs. Iterations
# -----------------------------------------------
plt.figure(figsize=(10, 6))
profit_values = -np.array(res.func_vals)
iterations = range(1, len(profit_values) + 1)
plt.plot(iterations, profit_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Profit per Batch (£)')
plt.title('Convergence of Bayesian Optimization')
plt.grid(True)

# Adjust y-axis limits based on profit values
lower_limit = np.percentile(profit_values, 5)
upper_limit = np.percentile(profit_values, 95)
margin = (upper_limit - lower_limit) * 0.05  # 5% margin
plt.ylim([lower_limit - margin, upper_limit + margin])

plt.show()

# Step 16: Visualization of Profit Function over Reflux Ratios and Distillate Flow Rates
# --------------------------------------------------------------------------------------
# Create a grid of reflux ratios and distillate flow rates
reflux_ratios_plot = np.linspace(reflux_ratio_min, reflux_ratio_max, 50)
distillate_flow_rates_plot = np.linspace(distillate_flow_rate_min, distillate_flow_rate_max, 50)
RR_grid, DFR_grid = np.meshgrid(reflux_ratios_plot, distillate_flow_rates_plot)

# Calculate profits over the grid
profit_grid = np.zeros_like(RR_grid)

for i in range(RR_grid.shape[0]):
    for j in range(RR_grid.shape[1]):
        params = [RR_grid[i, j], DFR_grid[i, j]]
        objective_val = cost_function(params)
        profit_grid[i, j] = -objective_val  # Convert back to profit

# Plot the profit contours
plt.figure(figsize=(12, 8))
contour = plt.contourf(RR_grid, DFR_grid, profit_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Profit per Batch (£)')
plt.scatter(
    optimal_reflux_ratio,
    optimal_distillate_flow_rate,
    color='red',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)
plt.xlabel('Reflux Ratio')
plt.ylabel('Distillate Flow Rate (L/h)')
plt.title('Profit Contour Plot over Reflux Ratio and Distillate Flow Rate')
plt.legend()
plt.grid(True)
plt.show()

# Step 17: Visualization of the Optimal Point with Respect to the Data
# --------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    reflux_ratio,
    Distillate_flowrate,
    -objective_values,  # Convert back to profit
    c=purities_percent,
    cmap='viridis',
    alpha=0.7,
    edgecolor='k',
    label='Experimental Data'
)

cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Methanol Purity (%)', rotation=270, labelpad=15)

ax.scatter(
    optimal_reflux_ratio,
    optimal_distillate_flow_rate,
    maximum_profit,
    color='red',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

ax.set_xlabel('Reflux Ratio')
ax.set_ylabel('Distillate Flow Rate (L/h)')
ax.set_zlabel('Profit per Batch (£)')
ax.set_title('Optimal Point in Relation to Experimental Data')

ax.legend()

plt.show()

# Step 18: Purity Surface Plot
# ----------------------------
# Create a grid of distillate and reflux flow rates
distillate_range = np.linspace(distillate_flow_rate_min, distillate_flow_rate_max, 50)
reflux_range = np.linspace(Reflux_flowrate.min(), Reflux_flowrate.max(), 50)
D_grid, R_grid = np.meshgrid(distillate_range, reflux_range)

# Predict purity over the grid
purity_grid = purity_function(D_grid, R_grid)

# Plot the purity surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    D_grid, R_grid, purity_grid,
    cmap='viridis',
    alpha=0.8,
    edgecolor='none'
)

# Overlay actual data points
ax.scatter(
    Distillate_flowrate,
    Reflux_flowrate,
    purities,
    color='red',
    label='Data Points',
    edgecolor='k'
)

ax.set_xlabel('Distillate Flow Rate (L/h)')
ax.set_ylabel('Reflux Flow Rate (L/h)')
ax.set_zlabel('Purity (Fraction)')
ax.set_title('Interpolated Purity Surface with Data Points')
ax.legend()

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Purity (Fraction)')
plt.show()

# Step 19: Energy Consumption Surface Plot
# ----------------------------------------
# Predict energy consumption over the grid
energy_grid = energy_function(D_grid, R_grid)

# Plot the energy consumption surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    D_grid, R_grid, energy_grid,
    cmap='plasma',
    alpha=0.8,
    edgecolor='none'
)

# Overlay actual data points
ax.scatter(
    Distillate_flowrate,
    Reflux_flowrate,
    energy_consumptions,
    color='green',
    label='Data Points',
    edgecolor='k'
)

ax.set_xlabel('Distillate Flow Rate (L/h)')
ax.set_ylabel('Reflux Flow Rate (L/h)')
ax.set_zlabel('Energy Consumption (kWh)')
ax.set_title('Interpolated Energy Consumption Surface with Data Points')
ax.legend()

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy Consumption (kWh)')
plt.show()
