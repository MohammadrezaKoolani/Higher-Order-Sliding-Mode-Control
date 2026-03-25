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
# Plots for tuning
# =====================

# 1) Tracking and vehicle states
plt.figure(figsize=(11, 7))
plt.suptitle("Tracking and Vehicle States")

plt.subplot(2, 2, 1)
plt.plot(t, e, label="e")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Lateral error [m]")
plt.title("Lateral Tracking Error")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, edot, label="edot")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Error rate [m/s]")
plt.title("Lateral Error Derivative")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, vy, label="vy")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Lateral velocity [m/s]")
plt.title("Lateral Velocity")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, r, label="r")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Yaw rate [rad/s]")
plt.title("Yaw Rate")
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 2) Sliding surface
plt.figure()
plt.plot(t, s_hist, label="s")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Sliding surface")
plt.title("Sliding Surface Evolution")
plt.legend()
plt.grid(True)
plt.show()

# 3) Steering decomposition
plt.figure()
plt.plot(t, delta_eq_hist, label="delta_eq")
plt.plot(t, delta_st_hist, label="delta_st")
plt.plot(t, delta_hist, label="delta")
plt.xlabel("Time [s]")
plt.ylabel("Steering [rad]")
plt.title("Steering Input Decomposition")
plt.legend()
plt.grid(True)
plt.show()

# 4) Internal super-twisting state
plt.figure()
plt.plot(t, u2_hist, label="u2")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("u2")
plt.title("Super-Twisting Internal State")
plt.legend()
plt.grid(True)
plt.show()