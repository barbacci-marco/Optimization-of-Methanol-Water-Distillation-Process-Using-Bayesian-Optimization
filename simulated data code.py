import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Number of data points to simulate
num_samples = 100

# Step 1: Generate random values for Reflux Flow Rate, Distillate Flow Rate, and Temperature within realistic ranges
# Assuming realistic ranges based on typical lab-scale distillation setups
reflux_flow_rate = np.random.uniform(0.5, 3.0, num_samples)         # L/h
distillate_flow_rate = np.random.uniform(1.0, 3.0, num_samples)     # L/h
temperature = np.random.uniform(94.0, 96.0, num_samples)            # °C

# Step 2: Calculate Reflux Ratio
reflux_ratio = reflux_flow_rate / distillate_flow_rate

# Step 3: Simulate Purity based on Reflux Ratio and Temperature
# Higher reflux ratio and optimal temperature generally lead to higher purity
def simulate_purity(reflux_ratio, temperature):
    # Temperature effect centered at 95°C
    temp_effect = -((temperature - 95.0) ** 2) / 0.5
    # Reflux ratio effect with diminishing returns
    reflux_effect = np.log(reflux_ratio + 1)
    # Combine effects with base purity
    purity = 0.9 + 0.05 * reflux_effect + 0.03 * np.exp(temp_effect)
    # Add random noise
    purity += np.random.normal(0, 0.005, size=reflux_ratio.shape)
    # Ensure purity is between 0 and 1
    purity = np.clip(purity, 0, 1)
    return purity

purity = simulate_purity(reflux_ratio, temperature)

# Step 4: Simulate Energy Consumption (Boiler Duty and Condenser Duty)
# Energy consumption increases with higher reflux flow rate and temperature deviations from optimal
def simulate_energy_consumption(reflux_flow_rate, temperature):
    # Base energy consumption
    boiler_duty = 500 + 200 * reflux_flow_rate  # Boiler duty increases with reflux flow rate
    condenser_duty = 400 + 150 * reflux_flow_rate  # Condenser duty also increases with reflux flow rate
    # Temperature penalty for deviations from 95°C
    temp_penalty = 50 * np.abs(temperature - 95.0)
    # Adjust duties with temperature penalty
    boiler_duty += temp_penalty
    condenser_duty += temp_penalty
    # Add random noise
    boiler_duty += np.random.normal(0, 10, size=boiler_duty.shape)
    condenser_duty += np.random.normal(0, 10, size=condenser_duty.shape)
    return boiler_duty, condenser_duty

boiler_duty, condenser_duty = simulate_energy_consumption(reflux_flow_rate, temperature)

# Step 5: Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Reflux Flow Rate': reflux_flow_rate,
    'Distillate Flow Rate': distillate_flow_rate,
    'Temperature': temperature,
    'Purity': purity,
    'Boiler Duty': boiler_duty,
    'Condenser Duty': condenser_duty
})

# Step 6: Save the DataFrame to an Excel file
data.to_csv('/Users/marcobarbacci/Y3 Labs BOPT /Year 3 distilattion optimisation/distillation_data.csv', index=False)

# Step 7: Optional - Visualize the simulated data
# Plot Purity vs Reflux Ratio
plt.figure(figsize=(8, 6))
plt.scatter(reflux_ratio, purity, c=temperature, cmap='viridis', edgecolor='k')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Reflux Ratio')
plt.ylabel('Purity')
plt.title('Simulated Purity vs Reflux Ratio')
plt.grid(True)
plt.show()

# Plot Energy Consumption vs Reflux Flow Rate
plt.figure(figsize=(8, 6))
total_energy = boiler_duty + condenser_duty
plt.scatter(reflux_flow_rate, total_energy, c=temperature, cmap='plasma', edgecolor='k')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Reflux Flow Rate (L/h)')
plt.ylabel('Total Energy Consumption (Watts)')
plt.title('Simulated Energy Consumption vs Reflux Flow Rate')
plt.grid(True)
plt.show()
