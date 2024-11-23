# Optimization-of-Methanol-Water-Distillation-Process-Using-Bayesian-Optimization

---
## Project Overview
This project focuses on optimizing a **methanol-water distillation process** using a **machine learning-driven approach**. By leveraging **Bayesian Optimization**, the goal is to identify the optimal operating conditions that minimize energy consumption while ensuring high product purity. Key operating variables include:
- **Reflux flow rate**
- **Distillate flow rate**
- **Column temperature**

Bayesian Optimization, a machine learning-based technique, is employed to efficiently explore the parameter space of this complex and expensive process, finding optimal operating points with minimal experimental evaluations. This approach integrates experimental data, predictive modeling, and cost-function optimization to achieve improved performance and efficiency.

---

## Project Workflow
### 1. **Data Collection**
Experimental data is collected from a lab-scale distillation rig, measuring:
- **Process variables**: Reflux flow rate, distillate flow rate, column temperature.
- **Outputs**: Product purity, energy consumption.

This data is organized in an Excel file for analysis.

### 2. **Interpolation Models**
Interpolation techniques are used to model the relationship between process variables and system outputs. These models predict:
- **Purity**
- **Energy consumption**
for any given set of operating points, enabling efficient evaluation of untested conditions.

### 3. **Cost Function Development**
A cost function is designed to quantify the performance of the distillation process by combining:
- **Energy consumption**
- **Product purity**

Penalties are applied to solutions that fail to meet the minimum product purity requirement, ensuring practical and feasible optimization.

### 4. **Bayesian Optimization**
Using the `gp_minimize` function from the **scikit-optimize** library, Bayesian Optimization is applied. This iterative algorithm:
- Balances **exploration** of new parameter combinations with **exploitation** of known optimal solutions.
- Minimizes the cost function while limiting the number of experimental evaluations.

### 5. **Analysis and Visualization**
The optimization results are analyzed and visualized to highlight:
- The effect of different operating conditions on process efficiency and purity.
- Comparisons between optimal conditions identified by Bayesian Optimization and traditional methods.

---

## Key Features
- **Machine Learning Integration**: Utilizes machine learning for optimizing complex, non-linear chemical processes.
- **Bayesian Optimization**: Reduces experimental time and resource usage by minimizing the number of required evaluations.
- **Data-Driven Approach**: Relies on real experimental data to build predictive models of the distillation process.
- **Economic Focus**: Balances energy consumption and product purity for cost-effective operation.

---

## Tools and Libraries
- **Python**: Programming language for scripting and implementation.
- **Pandas**: Data manipulation and handling, especially for reading Excel files.
- **Scikit-Optimize (skopt)**: Bayesian Optimization for parameter tuning.
- **Scikit-learn**: Interpolation models and additional machine learning tools.
- **Matplotlib**: Visualization of optimization results.

---

## Key Outcomes
This project demonstrates a modern, data-driven approach to chemical process optimization by integrating machine learning into traditional workflows. Benefits include:
- **Reduced experimental time** through efficient parameter search.
- **Improved process performance** by balancing energy costs with purity requirements.
- **Scalability** for industrial applications, offering a framework adaptable to larger-scale systems.
  
---
