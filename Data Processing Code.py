# Section 1: Import Required Libraries
# -------------------------------------
# Import necessary libraries for data manipulation, numerical analysis, and plotting.
# - NumPy for numerical operations
# - Pandas for data handling
# - Matplotlib for plotting (if required)
# - SciPy for interpolation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Section 2: Load and Extract Calibration Data from CSV
# ------------------------------------------------------
# Load calibration data from CSV, including methanol density and purity data,
# along with operational data for temperature, flow rates, and condenser information.
# Assumes CSV columns for sample number, methanol %, densities, temperatures, and flow rates.
data = pd.read_csv('Methanol_density.csv')

# Extract data columns for analysis.
sample_number = data['Sample number'].values                 # Sample identifiers
Methanol_per_cal = data['Methanol%'].values                  # Calibration methanol percentage
Methanol_den_cal = data['Density (g·cm-3)'].values           # Density for calibration of methanol purity
Methanol_den_real = data['Sample density(g·cm-3)'].values    # Density values for purity interpolation
WCondensor_T_Out = data['Condensor T out'].values            # Condenser outlet temperature
WCondensor_T_in = data['Condensor T in '].values             # Condenser inlet temperature
Avg_Temp = data['average temperature'].values                # Average temperature in Celsius
Dist_flowrate = data['Distilate flowrate'].values            # Distillate flow rate
R_flowrate = data['reflux flow rate'].values                 # Reflux flow rate
vol_flow_rate_W = data['Flowrate of water'].values           # Water flow rate for condenser

# Convert average temperature to Kelvin
Avg_Temp_K = Avg_Temp + 273.15  # Kelvin conversion for average temperature
cp_W = 4200  # J/kg·K, specific heat capacity of water

# Section 3: Methanol Purity Interpolation Setup
# -----------------------------------------------
# Create an interpolator to convert sample density values into methanol percentages
# based on calibration data for purity determination. Linear extrapolation extends
# the interpolation range as needed.
purity_interpolator = interp1d(Methanol_den_cal, Methanol_per_cal, kind='linear', fill_value="extrapolate")

# Apply interpolator to sample densities to obtain methanol purity percentages
Purity = purity_interpolator(Methanol_den_real)

# Section 5: Calculate Heat Transfer for Condenser and Reboiler
# -------------------------------------------------------------
# Calculate heat transfer rates for the condenser and reboiler:
# - Condenser Duty (water cooling) based on water flow rate, specific heat, and temperature change
# - Reboiler Duty is set to a fixed output of 3 kW, with duty expressed in kilowatt-hours for a 1-hour operation

# Condenser Duty calculation (in kW for a 1-hour operation period)
Condenser_Duty = (vol_flow_rate_W * cp_W * (WCondensor_T_Out - WCondensor_T_in) * 60 / 1000)

# Reboiler Duty is fixed at 3 kW, with duty in kW·h
Reboiler_Duty = 3 * 60  # kW·h for a 1-hour operation period

# Section 6: Save Output Data for Analysis
# -----------------------------------------
# Organize the data into a DataFrame, including sample density, interpolated methanol purity,
# and operational data such as heat duties, reboiler temperature, and flow rates.
processed_data = pd.DataFrame({
    'Sample Number': sample_number,
    'Sample Density (g·cm^−3)': Methanol_den_real,  # Actual density values for each sample
    'Methanol Purity (%)': Purity,                   # Interpolated methanol purity from density
    'Water Cooling Heat Transfer (kW/h)': Condenser_Duty,  # Condenser duty in kW·h
    'Electric Reboiler Duty (kW/h)': Reboiler_Duty,        # Fixed reboiler duty in kW·h
    'Reboiler Temperature (K)': Avg_Temp_K,                # Average reboiler temperature in Kelvin
    'Distillate Flowrate (ml/h)': Dist_flowrate,           # Distillate flow rate in ml/h
    'Reflux Flowrate (ml/h)': R_flowrate,                  # Reflux flow rate in ml/h
})

# Export DataFrame to CSV for record-keeping and external analysis
processed_data.to_csv('Lab_distillation_data.csv', index=False)

# Section 7: Display Results
# --------------------------
# Display the resulting DataFrame to verify the data and purity interpolation.
print("\nProcessed Data:")
print(output_data)
