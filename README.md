# Higher-Order Sliding Mode Control for Heavy-Truck Lateral Dynamics

This repository contains my implementation of a **higher-order sliding mode controller (HOSM)** for **lateral path-tracking control** using two different vehicle dynamics models:

1. a **simple bicycle model** with a single steering input,
2. a more detailed **8-wheel heavy-truck model** with **two front steering axles**.

The main goal of this project is to compare the closed-loop responses of the two models under the same control philosophy and study how model complexity affects tracking performance, yaw behavior, and steering effort.

---

## Overview

The controller is based on the **super-twisting higher-order sliding mode algorithm**, implemented for lateral dynamics tracking.

The repository includes two simulation scripts:

- `higher_order_SMC.py`  
  Higher-order sliding mode control applied to a **4-wheel / bicycle-model** representation of the truck.

- `higher_order_smc_mu_ax_v.py`  
  Higher-order sliding mode control applied to an **8-wheel heavy-truck model** with **dual front-axle steering**.

Both scripts simulate lateral motion on a curved road and generate plots for response analysis.

---

## Motivation

Heavy trucks can be modeled at different levels of complexity. A simple bicycle model is easier to analyze and control, but it may not capture the behavior of a real multi-axle vehicle accurately enough. On the other hand, a more detailed multi-axle model better represents the actual dynamics, but it is more complex.

This project was developed to compare these two modeling approaches under a higher-order sliding mode controller and investigate:

- lateral tracking performance,
- yaw-rate response,
- auxiliary lateral/heading behavior,
- steering effort,
- sliding-surface convergence.

---

## Implemented Models

### 1) Bicycle Model (`higher_order_SMC.py`)

This script implements a reduced-order lateral dynamics model with:

- lateral tracking error `e`
- derivative of lateral error `edot`
- lateral velocity `vy`
- yaw rate `r`
- yaw angle `psi`

Main characteristics:

- single steering input,
- front/rear cornering stiffness lumped into a bicycle model,
- constant longitudinal speed,
- constant-curvature road,
- super-twisting HOSM controller with steering saturation.

The sliding variable is defined as:

```math
s = \dot{e} + \lambda e
```

The controller is composed of:

- an **equivalent control** term,
- a **super-twisting correction** term.

---

### 2) 8-Wheel Heavy-Truck Model (`higher_order_smc_mu_ax_v.py`)

This script implements a more detailed lateral path-tracking model for a heavy multi-axle vehicle.

State vector:

```math
x = [v_y,\ r,\ e_{dL},\ e_{\phi L}]^T
```

where:

- `vy` = lateral velocity,
- `r` = yaw rate,
- `e_dL` = lateral displacement error,
- `e_phiL` = heading error.

Main characteristics:

- 8-wheel heavy-truck representation,
- two front steering axles,
- steering relation between the first and second front axles,
- nominal and perturbed plant matrices,
- steady-state reference generation for curved-road motion,
- LQR-based sliding surface design,
- super-twisting HOSM law applied to tracking error coordinates.

For this model, the controller is applied to the tracking error:

```math
z = x - x_{ref}
```

and the sliding variable is defined as:

```math
s = Cz
```

where `C` is obtained from a continuous-time Riccati equation.

---

## Control Strategy

Both implementations use the **super-twisting algorithm**, a second-order sliding mode method designed to reduce the chattering usually associated with classical first-order sliding mode control.

The control structure is:

1. define a sliding variable,
2. compute an equivalent control term,
3. add the super-twisting corrective term,
4. apply steering saturation.

The super-twisting part follows the form:

```math
u_1 = -\alpha \sqrt{|s|}\,\mathrm{sign}(s)
```

```math
\dot{u}_2 = -\beta\,\mathrm{sign}(s)
```

with the total corrective action built from `u1` and `u2`.

---

## Simulation Setup

The current scripts use:

- simulation time: **50 s**
- time step: **0.01 s**
- constant forward speed: **13.5 m/s**
- constant road curvature corresponding to **R = 50 m**

### Initial conditions

#### Bicycle model
- initial lateral error: `e(0) = 1.0 m`

#### 8-wheel model
- initial lateral error: `e_dL(0) = 1.0 m`
- initial heading error: `e_phiL(0) = 0.05 rad`

### Steering limits

- bicycle model: `delta_max = 0.5 rad`
- 8-wheel model: `delta_max = 20 deg`

---

## Outputs

The scripts generate plots showing:

- lateral error response,
- yaw-rate response,
- auxiliary lateral/heading state response,
- steering input,
- sliding-surface evolution,
- equivalent and super-twisting control components.

They also print numerical performance indicators in the console, such as:

- final tracking states,
- maximum steering magnitude,
- RMS tracking error,
- final sliding variable.

---

## Repository Structure

```text
.
├── higher_order_SMC.py
├── higher_order_smc_mu_ax_v.py
├── Higher-order_sliding_mode_control_for_lateral_dynamics_of_autonomous_vehicles_with_experimental_validation.pdf
└── 1-s2.0-S0957415825001060-main.pdf
```

---

## Requirements

The code is written in Python and uses the following libraries:

- `numpy`
- `matplotlib`
- `scipy` *(required for the 8-wheel model script)*

Install dependencies with:

```bash
pip install numpy matplotlib scipy
```

---

## How to Run

Run the bicycle-model simulation:

```bash
python higher_order_SMC.py
```

Run the 8-wheel heavy-truck simulation:

```bash
python higher_order_smc_mu_ax_v.py
```

---

## Comparison Note

This repository is intended to compare the responses of two different vehicle models under a higher-order sliding mode control framework.

At the moment, the comparison is mainly **qualitative**, based on:

- separate closed-loop response plots,
- printed simulation metrics,
- differences in steering demand and tracking behavior.

Since the two models do not use exactly the same state definitions, reference construction, and steering architecture, this should be interpreted as a **model-based comparison study**, rather than a perfectly unified benchmark.

A natural next step would be to add:

- common performance indices,
- overlay plots for both models,
- settling-time comparison,
- control-energy comparison,
- robustness tests with noise and disturbances.

---

## References

### Higher-order sliding mode control paper
G. Tagne, R. Talj, and A. Charara,  
**“Higher-Order Sliding Mode Control for Lateral Dynamics of Autonomous Vehicles, with Experimental Validation.”**  
IEEE Intelligent Vehicles Symposium (IV), 2013.  
DOI: [10.1109/IVS.2013.6629545](https://doi.org/10.1109/IVS.2013.6629545)

### Multi-axle vehicle dynamics paper
L. Guo, J. Zhao, L. Guan, J. Wang, P. Ge, and L. Xu,  
**“Coordination control of multi-axle distributed drive vehicle with dynamically-triggered DYC intervention and KKT-based torque optimization distribution.”**  
*Mechatronics*, 111 (2025), 103397.  
DOI: [10.1016/j.mechatronics.2025.103397](https://doi.org/10.1016/j.mechatronics.2025.103397)

---

## Acknowledgment

This project combines a higher-order sliding mode lateral controller from the literature with two different vehicle dynamics representations in order to study the influence of model complexity on heavy-truck lateral control performance.
