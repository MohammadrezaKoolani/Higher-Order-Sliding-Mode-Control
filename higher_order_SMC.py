import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# =====================
# Simulation settings
# =====================
T = 50.0
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
# Unified plots for comparison
# =====================

plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",          # default text
    "axes.titlesize": 16,
    "axes.titleweight": "bold",     # subplot titles
    "axes.labelsize": 15,
    "axes.labelweight": "bold",     # x/y labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# Reference signals
e_ref = np.zeros_like(t)
r_ref = np.full_like(t, Vx / R)   # steady yaw-rate reference for constant curvature
aux_ref = np.zeros_like(t)        # auxiliary state reference

# 1) Main comparison figure: same structure for both models
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Closed-Loop Lateral Dynamics Comparison(4 Wheels)", fontsize=22, fontweight="bold")

# (1,1) Lateral error
axs[0, 0].plot(t, e, linewidth=2.2, label="Actual")
axs[0, 0].plot(t, e_ref, "--", linewidth=2.0, label="Reference")
axs[0, 0].set_title("Lateral Error")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Error [m]")
axs[0, 0].grid(True)
axs[0, 0].legend()

# (1,2) Yaw rate
axs[0, 1].plot(t, r, linewidth=2.2, label="Actual")
axs[0, 1].plot(t, r_ref, "--", linewidth=2.0, label="Reference")
axs[0, 1].set_title("Yaw Rate")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Yaw rate [rad/s]")
axs[0, 1].grid(True)
axs[0, 1].legend()

# (2,1) Auxiliary lateral state
axs[1, 0].plot(t, vy, linewidth=2.2, label="Actual")
axs[1, 0].plot(t, aux_ref, "--", linewidth=2.0, label="Reference")
axs[1, 0].set_title("Auxiliary Lateral State")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Lateral velocity [m/s]")
axs[1, 0].grid(True)
axs[1, 0].legend()

# (2,2) Steering input
axs[1, 1].plot(t, delta_hist, linewidth=2.2, label="Steering input")
axs[1, 1].set_title("Steering Input")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Steering angle [rad]")
axs[1, 1].grid(True)
axs[1, 1].legend()

for ax in axs.flat:
    ax.set_xlim(0, 50)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# 2) Sliding surface
plt.figure(figsize=(8, 4.5))
plt.plot(t, s_hist, linewidth=3.0, label="Actual")
plt.plot(t, np.zeros_like(t), "--", linewidth=2.5, label="Reference")
plt.title("Sliding Surface (4 Wheels)", fontsize=22, fontweight="bold")
plt.xlabel("Time [s]")
plt.ylabel("s")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# 3) Control components
plt.figure(figsize=(10, 4.5))
plt.plot(t, delta_eq_hist, linewidth=2.2, label="Equivalent component")
plt.plot(t, delta_st_hist, linewidth=2.2, label="Super-twisting component")
plt.title("Steering Control Components", fontsize=14, fontweight="bold")
plt.xlabel("Time [s]")
plt.ylabel("Steering angle [rad]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()