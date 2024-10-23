import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Set a random seed for reproducibility
np.random.seed(42)

# Number of data points to simulate
num_samples = 200

# Generate random values for Reflux Flow Rate, Distillate Flow Rate, and Temperature within realistic ranges
# Assuming realistic ranges based on typical lab-scale distillation setups
reflux_flow_rate = np.random.uniform(0.5, 10.0, num_samples)         # L/h
distillate_flow_rate = np.random.uniform(1.0, 10.0, num_samples)     # L/h
temperature = np.random.uniform(94.0, 96.0, num_samples)
reflux_ratios = reflux_flow_rate / distillate_flow_rate

# Constants and Parameters
alpha = 2.0    # Relative volatility of methanol to water
zF = 0.30      # Feed composition (methanol mole fraction)
xB = 0.05      # Given bottoms composition
N_stages = 8   # Total number of stages (including reboiler)
NF = 4         # Feed stage location (from top)
q = 1.0        # Saturated liquid feed (q = 1)

# Equilibrium curve function
def y_eq(x):
    return alpha * x / (1 + (alpha - 1) * x)

# Function to perform stage stepping and count stages
def count_stages(xD, R):
    # Operating line parameters
    L_over_V = R / (R + 1)
    m_r = L_over_V
    b_r = xD / (R + 1)

    # For q = 1 (saturated liquid feed), the q-line is vertical at x = zF
    x_q = zF
    y_q = m_r * x_q + b_r  # Intersection of rectifying line and q-line

    # Stripping line parameters
    m_s = (y_q - xB) / (x_q - xB)
    b_s = xB - m_s * xB

    # Initialize variables
    stages = 0
    x = xD
    max_iterations = 500  # Increased to allow convergence

    while x > xB + 1e-6 and stages < max_iterations:
        # Equilibrium step (vertical line)
        y = y_eq(x)

        # Operating line step (horizontal line)
        if stages < NF - 1:
            # Rectifying section
            x_new = (y - b_r) / m_r
        else:
            # Stripping section
            x_new = (y - b_s) / m_s

        # Prevent infinite loop due to numerical errors
        if abs(x_new - x) < 1e-6 or x_new < xB - 1e-6 or x_new > xD + 1e-6:
            break

        x = x_new
        stages += 1

    return stages

# Function to find xD for a given R
def find_xD(R):
    # Objective function to match the number of stages
    def objective(xD_array):
        xD = xD_array[0]
        if xD <= xB + 1e-6 or xD >= 1.0 - 1e-6:
            return 10.0  # Large error if xD is out of bounds
        stages = count_stages(xD, R)
        return stages - N_stages

    # Improved initial guess for xD
    xD_initial = xB + (1 - xB) * (R / (R + 1))

    try:
        # Solve for xD
        xD_solution = fsolve(objective, xD_initial, xtol=1e-6)
        xD = xD_solution[0]

        # Validate solution
        if xD <= xB + 1e-6 or xD >= 1.0 - 1e-6:
            return None
        return xD
    except RuntimeError:
        # Solver failed to converge
        return None

# Lists to store results
purity_list = []
reflux_ratio_list = []

for R in reflux_ratios:
    xD = find_xD(R)
    if xD is not None and not np.isnan(xD):
        purity_list.append(xD)
        reflux_ratio_list.append(R)
    else:
        print(f"Warning: No valid xD found for R = {R}")

# Ensure that data lists are of the same length
if len(purity_list) != len(reflux_ratio_list):
    min_length = min(len(purity_list), len(reflux_ratio_list))
    purity_list = purity_list[:min_length]
    reflux_ratio_list = reflux_ratio_list[:min_length]

# Create DataFrame
data = pd.DataFrame({
    'Reflux Ratio': reflux_ratio_list,
    'Distillate Purity': purity_list
})

# Save to CSV
data.to_csv('distillation_data.csv', index=False)

# Display data
print("\nGenerated Data:")
print(data)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(data['Reflux Ratio'], data['Distillate Purity'], marker='o')
plt.xlabel('Reflux Ratio (R)')
plt.ylabel('Distillate Purity (xD)')
plt.title('Distillate Purity vs. Reflux Ratio at Fixed Number of Stages')
plt.grid(True)
plt.show()

# Fit a trend line (e.g., polynomial fit)
coefficients = np.polyfit(data['Reflux Ratio'], data['Distillate Purity'], deg=3)
trendline = np.poly1d(coefficients)

# Generate additional reflux ratios within the original range
additional_R = np.linspace(reflux_ratios.min(), reflux_ratios.max(), 200)

# Calculate the trendline distillate purities for these reflux ratios
trend_purities = trendline(additional_R)

# Generate random data points within ±10% error range of the trendline
error_range = 0.10  # 10% error range

# Calculate the lower and upper bounds
lower_bounds = trend_purities * (1 - error_range)
upper_bounds = trend_purities * (1 + error_range)

# Generate random purities within the bounds
random_purities = np.random.uniform(lower_bounds, upper_bounds)

# Step 4: Simulate Energy Consumption (Boiler Duty and Condenser Duty)
# Generate new distillate flow rates and temperatures matching the length of additional_R
num_generated_samples = len(additional_R)  # 100
distillate_flow_rate_generated = np.random.uniform(1.0, 3.0, num_generated_samples)
temperature_generated = np.random.uniform(94.0, 96.0, num_generated_samples)

# Calculate corresponding reflux flow rates
reflux_flow_rate_generated = additional_R * distillate_flow_rate_generated

# Simulate energy consumption
def simulate_energy_consumption(reflux_flow_rate):
    # Define exponential trendline for total energy consumption
    E0 = 1000  # base value
    k = 0.1    # exponent coefficient
    E_total_trend = E0 * np.exp(k * reflux_flow_rate)
    # Generate data points within ±10% error margin
    error_margin = 0.50
    lower_bounds = E_total_trend * (1 - error_margin)
    upper_bounds = E_total_trend * (1 + error_margin)
    E_total_generated = np.random.uniform(lower_bounds, upper_bounds)
    # Split total energy into boiler and condenser duties (e.g., 60% and 40%)
    boiler_duty = E_total_generated * 0.6
    condenser_duty = E_total_generated * 0.4
    return boiler_duty, condenser_duty

boiler_duty_generated, condenser_duty_generated = simulate_energy_consumption(reflux_flow_rate_generated)

# Create a DataFrame for the generated data points
generated_data = pd.DataFrame({
    'Reflux Ratio': additional_R,
    'Distillate Purity': random_purities,
    'Reflux Flow Rate': reflux_flow_rate_generated,
    'Distillate Flow Rate': distillate_flow_rate_generated,
    'Temperature': temperature_generated,
    'Boiler Duty': boiler_duty_generated,
    'Condenser Duty': condenser_duty_generated
})

# Save generated data to CSV
generated_data.to_csv('distillation_generated_data.csv', index=False)

# Plot the generated data points
plt.scatter(generated_data['Reflux Ratio'], generated_data['Distillate Purity'], color='red', alpha=0.5, label='Generated Data')
plt.xlabel('Reflux Ratio (R)')
plt.ylabel('Distillate Purity (xD)')
plt.title('Generated Distillate Purity vs. Reflux Ratio')
plt.legend()
plt.show()

# Plot Energy Consumption vs Reflux Flow Rate
plt.figure(figsize=(8, 6))
total_energy = generated_data['Boiler Duty'] + generated_data['Condenser Duty']
plt.scatter(generated_data['Reflux Flow Rate'], total_energy, c=generated_data['Temperature'], cmap='plasma', edgecolor='k')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Reflux Flow Rate (L/h)')
plt.ylabel('Total Energy Consumption (Units)')
plt.title('Simulated Energy Consumption vs Reflux Flow Rate')
plt.grid(True)
plt.show()
