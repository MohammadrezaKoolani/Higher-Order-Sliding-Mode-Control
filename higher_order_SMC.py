import numpy as np
import matplotlib.pyplot as plt

# =====================
# Simulation settings
# =====================
T = 20.0
dt = 0.01
N = int(T / dt) + 1
t = np.linspace(0, T, N)

# =====================
# States
# =====================
e = np.zeros(N)       # lateral tracking error
edot = np.zeros(N)    # derivative of lateral tracking error
vy = np.zeros(N)      # lateral velocity
r = np.zeros(N)       # yaw rate
psi = np.zeros(N)     # yaw angle

# =====================
# Control histories
# =====================
s_hist = np.zeros(N)
delta_hist = np.zeros(N)
delta_eq_hist = np.zeros(N)
delta_st_hist = np.zeros(N)
u2_hist = np.zeros(N)

# Initial conditions
e[0] = 1.0
u2 = 0.0

# =====================
# Vehicle parameters
# =====================
m = 25000.0
Cf = 175000.0
Cr = 173600.0
Iz = 98000.0
Lf = 1.20
Lr = 1.20
Vx = 13.5

# =====================
# Control parameters
# =====================
lam = 8.0
alpha = 0.1
beta = 0.03

# Optional steering limit
delta_max = 0.5  # rad

# =====================
# Road curvature
# =====================
R = 50.0

def road_curvature(time):
    return 1.0 / R

# =====================
# Simulation loop
# =====================
for k in range(N - 1):
    curv = road_curvature(t[k])

    # Sliding surface
    s = edot[k] + lam * e[k]
    s_hist[k] = s
    u2_hist[k] = u2

    # Equivalent control term
    phi = (-(Cf + Cr) / (m * Vx)) * vy[k] \
          - ((Lf * Cf - Lr * Cr) / (m * Vx)) * r[k] \
          - Vx**2 * curv \
          + lam * edot[k]

    delta_eq = -(m / Cf) * phi

    # Super-twisting term
    u1 = -alpha * np.sqrt(abs(s) + 1e-12) * np.sign(s)
    delta_st = u1 + u2

    # Total steering
    delta = delta_eq + delta_st
    delta = np.clip(delta, -delta_max, delta_max)

    # Store controls
    delta_eq_hist[k] = delta_eq
    delta_st_hist[k] = delta_st
    delta_hist[k] = delta

    # Explicit Euler update of u2
    u2 += dt * (-beta * np.sign(s))

    # Vehicle model
    vy_dot = (-(Cf + Cr) / (m * Vx)) * vy[k] \
             - (((Lf * Cf - Lr * Cr) / (m * Vx)) + Vx) * r[k] \
             + (Cf / m) * delta

    r_dot = (-(Lf * Cf - Lr * Cr) / (Iz * Vx)) * vy[k] \
            - ((Lf**2 * Cf + Lr**2 * Cr) / (Iz * Vx)) * r[k] \
            + (Lf * Cf / Iz) * delta

    # Tracking-error dynamics
    e_ddot = (-(Cf + Cr) / (m * Vx)) * vy[k] \
             - ((Lf * Cf - Lr * Cr) / (m * Vx)) * r[k] \
             - Vx**2 * curv \
             + (Cf / m) * delta

    # State update
    vy[k+1] = vy[k] + dt * vy_dot
    r[k+1] = r[k] + dt * r_dot
    psi[k+1] = psi[k] + dt * r[k]

    edot[k+1] = edot[k] + dt * e_ddot
    e[k+1] = e[k] + dt * edot[k]

# =====================
# Final sample
# =====================
s_hist[-1] = edot[-1] + lam * e[-1]
delta_hist[-1] = delta_hist[-2]
delta_eq_hist[-1] = delta_eq_hist[-2]
delta_st_hist[-1] = delta_st_hist[-2]
u2_hist[-1] = u2

# =====================
# Performance metrics
# =====================
e_rms = np.sqrt(np.mean(e**2))

print("Final e         =", e[-1])
print("Final edot      =", edot[-1])
print("Final vy        =", vy[-1])
print("Final r         =", r[-1])
print("Final psi       =", psi[-1])
print("Final s         =", s_hist[-1])
print("Max |e|         =", np.max(np.abs(e)))
print("RMS(e)          =", e_rms)
print("Max |delta|     =", np.max(np.abs(delta_hist)))
print("Final u2        =", u2_hist[-1])

# =====================
# Plots
# =====================

# =====================
# Plots
# =====================

# Reference signals for regulation
e_ref = np.zeros_like(t)
edot_ref = np.zeros_like(t)

# 1) Tracking error states
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Tracking Error Dynamics", fontsize=14)

# Lateral tracking error
axs[0, 0].plot(t, e, label="e")
axs[0, 0].plot(t, e_ref, "--", label="e_ref")
axs[0, 0].set_title("Lateral Tracking Error")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("e [m]")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Derivative of tracking error
axs[0, 1].plot(t, edot, label="edot")
axs[0, 1].plot(t, edot_ref, "--", label="edot_ref")
axs[0, 1].set_title("Tracking Error Derivative")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("edot [m/s]")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Lateral velocity
axs[1, 0].plot(t, vy, label="vy")
axs[1, 0].set_title("Lateral Velocity")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("vy [m/s]")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Yaw rate
axs[1, 1].plot(t, r, label="r")
axs[1, 1].set_title("Yaw Rate")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("r [rad/s]")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 2) Yaw angle and sliding variable
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# fig.suptitle("Yaw Motion and Sliding Variable", fontsize=14)

# # Yaw angle
# axs[0].plot(t, psi, label="psi")
# axs[0].set_title("Yaw Angle")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("psi [rad]")
# axs[0].grid(True)
# axs[0].legend()

# Sliding surface
plt.figure(figsize=(3, 3))  # smaller figure
plt.plot(t, s_hist, label="s", linewidth=2.5)                 # thicker line
plt.plot(t, np.zeros_like(t), "--", label="s_ref", linewidth=2)
plt.title("Sliding Surface", fontsize=16, fontweight="bold") # bigger title
plt.xlabel("Time [s]", fontsize=18, fontweight="bold")
plt.ylabel("s = edot + lambda e", fontsize=18, fontweight="bold")
plt.legend(fontsize=11)
plt.tight_layout()
plt.grid(True)
plt.show()


# 3) Steering input
plt.figure(figsize=(10, 4))
plt.plot(t, delta_hist, label="delta")
plt.title("Steering Input")
plt.xlabel("Time [s]")
plt.ylabel("Steering angle [rad]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 4) Steering control components
plt.figure(figsize=(10, 4))
plt.plot(t, delta_eq_hist, label="delta_eq")
plt.plot(t, delta_st_hist, label="delta_st")
plt.title("Steering Control Components")
plt.xlabel("Time [s]")
plt.ylabel("Steering angle [rad]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 5) Internal super-twisting state
plt.figure(figsize=(10, 4))
plt.plot(t, u2_hist, label="u2")
plt.title("Super-Twisting Internal State")
plt.xlabel("Time [s]")
plt.ylabel("u2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()