import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# ====================
# Simulation Settings
# ====================
T = 20.0
dt = 0.01
N = int(T / dt) + 1
t = np.linspace(0.0, T, N)

# ===================
# Truck Parameters
# ===================
m = 25000.0
Iz = 98000.0

L1 =  2.40
L2 =  0.68
L3 = -0.68
L4 = -2.40

Ca1 = 175000.0
Ca2 = 175000.0
Ca3 = 173600.0
Ca4 = 173600.0

Vx = 13.5
ld = 5.0

# Perturbed Plant Parameters
m_p  = 1.0 * m
Iz_p = 1.0 * Iz

Ca1_p = 1.0 * Ca1
Ca2_p = 1.0 * Ca2
Ca3_p = 1.0 * Ca3
Ca4_p = 1.0 * Ca4

Vx_p = 1.0 * Vx


# dual-front-steering relation
a_s = (2.0 * L2 - L3 - L4) / (2.0 * L1 - L3 - L4)

# =========================
# Linearized truck model
# =========================
sumCa   = Ca1 + Ca2 + Ca3 + Ca4
sumCaL  = Ca1 * L1 + Ca2 * L2 + Ca3 * L3 + Ca4 * L4
sumCaL2 = Ca1 * L1**2 + Ca2 * L2**2 + Ca3 * L3**2 + Ca4 * L4**2

a11 = -sumCa / (m * Vx)
a12 = -sumCaL / (m * Vx) - Vx
a21 = -sumCaL / (Iz * Vx)
a22 = -sumCaL2 / (Iz * Vx)

b1 = (Ca1 + a_s * Ca2) / m
b2 = (Ca1 * L1 + a_s * Ca2 * L2) / Iz

# x = [vy, r, e_dL, e_phiL]^T
A = np.array([
    [a11, a12, 0.0, 0.0],
    [a21, a22, 0.0, 0.0],
    [-1.0, -ld, 0.0, Vx],
    [0.0,  1.0, 0.0, 0.0]
], dtype=float)

B = np.array([
    [b1],
    [b2],
    [0.0],
    [0.0]
], dtype=float)

# curvature enters e_phiL_dot = r - Vx*kappa
E = np.array([
    [0.0],
    [0.0],
    [0.0],
    [-Vx]
], dtype=float)


# Perturbed plant model
sumCa_p   = Ca1_p + Ca2_p + Ca3_p + Ca4_p
sumCaL_p  = Ca1_p * L1 + Ca2_p * L2 + Ca3_p * L3 + Ca4_p * L4
sumCaL2_p = Ca1_p * L1**2 + Ca2_p * L2**2 + Ca3_p * L3**2 + Ca4_p * L4**2

a11_p = -sumCa_p / (m_p * Vx_p)
a12_p = -sumCaL_p / (m_p * Vx_p) - Vx_p
a21_p = -sumCaL_p / (Iz_p * Vx_p)
a22_p = -sumCaL2_p / (Iz_p * Vx_p)

b1_p = (Ca1_p + a_s * Ca2_p) / m_p
b2_p = (Ca1_p * L1 + a_s * Ca2_p * L2) / Iz_p

A_p = np.array([
    [a11_p, a12_p, 0.0, 0.0],
    [a21_p, a22_p, 0.0, 0.0],
    [-1.0, -ld, 0.0, Vx_p],
    [0.0,  1.0, 0.0, 0.0]
], dtype=float)

B_p = np.array([
    [b1_p],
    [b2_p],
    [0.0],
    [0.0]
], dtype=float)

E_p = np.array([
    [0.0],
    [0.0],
    [0.0],
    [-Vx_p]
], dtype=float)


# =====================
# States
# =====================
vy = np.zeros(N)
r = np.zeros(N)
e_dL = np.zeros(N)
e_phiL = np.zeros(N)

e_dL[0] = 2.0
e_phiL[0] = 0.05

# ============================================
# HOSM controller on z = x - x_ref(kappa)
# ============================================
# Derived sliding surface:
# s = -vy_tilde - ld*r_tilde + lam*e_dL_tilde + Vx*e_phiL_tilde
# C = np.array([[-3.67746041, 14.76200588, 7.88453846, 11.85602271]], dtype=float)
Q = np.diag([0.2, 0.40, 80.0, 18.0])
R_lqr = np.array([[1.5]])

P = solve_continuous_are(A, B, Q, R_lqr)
C = (B.T @ P).astype(float)   # 1x4 row vector

Gamma = float((C @ B).item())
if abs(Gamma) < 1e-9:
    raise ValueError("Gamma = C @ B is zero. Choose another sliding surface.")

alpha = 0.75
beta = 0.08
u2 = 0.0

# Disturbance parameters
d1 = 0.0 #0.10      # disturbance in vy_dot [m/s^2]
d2 = 0.0 #0.02      # disturbance in r_dot [rad/s^2]
d3 = 0.0 #0.02      # disturbance in e_dL_dot [m/s]
d4 = 0.0 #0.005     # disturbance in e_phiL_dot [rad/s]

delta_max = np.deg2rad(20.0)

# histories
s_hist = np.zeros(N)
delta1_hist = np.zeros(N)
delta2_hist = np.zeros(N)
delta_ref_hist = np.zeros(N)
r_ref_hist = np.zeros(N)
ephi_ref_hist = np.zeros(N)
z_hist = np.zeros((N, 4))
x_ref_hist = np.zeros((N, 4))
delta_eq_hist = np.zeros(N)
delta_st_hist = np.zeros(N)
r_meas_hist = np.zeros(N)

R = 50.0
def road_curvature(time):
    return 1.0 / R


def steady_state_reference(kappa):
    """
    Compute x_ref(kappa) and delta_ref(kappa) from the nominal steady turn:
        0 = A x_ref + B delta_ref + E kappa
    with:
        e_dL_ref = 0
        r_ref    = Vx * kappa
    """
    r_ref = Vx_p * kappa

    # Solve the steady-state lateral/yaw equations for vy_ref and delta_ref
    M = np.array([
        [a11, b1],
        [a21, b2]
    ], dtype=float)

    rhs = np.array([
        -a12 * r_ref,
        -a22 * r_ref
    ], dtype=float)

    vy_ref, delta_ref = np.linalg.solve(M, rhs)

    e_dL_ref = 0.0
    e_phiL_ref = (vy_ref + ld * r_ref) / Vx

    x_ref = np.array([vy_ref, r_ref, e_dL_ref, e_phiL_ref], dtype=float)
    return x_ref, float(delta_ref)

# Creating Measurement Noise
np.random.seed(1)
sigma_r = 0.0 #0.05 / 3.0 # about 99.7% of samples within +- 0.05 rad/s

x_ref_prev = None

for k in range(N - 1):
    kappa = road_curvature(t[k])
    # x = np.array([vy[k], r[k], e_dL[k], e_phiL[k]], dtype=float)
    x_true = np.array([vy[k], r[k], e_dL[k], e_phiL[k]], dtype=float)
    r_meas = r[k] + np.random.normal(0.0, sigma_r)
    x_meas = np.array([vy[k], r_meas, e_dL[k], e_phiL[k]], dtype=float)

    # current reference
    x_ref, delta_ref = steady_state_reference(kappa)

    # finite-difference x_ref_dot
    # for constant curvature, this becomes zero after the first step
    if x_ref_prev is None:
        x_ref_dot = np.zeros(4)
    else:
        x_ref_dot = (x_ref - x_ref_prev) / dt

    # z dynamics:
    # z_dot = A z + B u + d_ref
    # where u = delta1 - delta_ref
    d_ref = A @ x_ref + B[:, 0] * delta_ref + E[:, 0] * kappa - x_ref_dot

    z = x_meas - x_ref

    # sliding variable on tracking-error state
    s = float((C @ z).item())

    # s_dot = C(A z + d_ref) + C B u
    Phi = float((C @ (A @ z + d_ref)).item())

    # equivalent control in error coordinates
    u_eq = -Phi / Gamma

    # super-twisting correction
    u1 = -alpha * np.sqrt(abs(s) + 1e-12) * np.sign(s)
    u2 += dt * (-beta * np.sign(s))
    u_st = (u1 + u2) / Gamma

    # actual front-axle steering
    delta1 = np.clip(delta_ref + u_eq + u_st, -delta_max, delta_max)

    # second front axle
    delta2 = a_s * delta1

    # plant propagation
    w = np.array([d1, d2, d3, d4], dtype=float) # Disturbance vector
    # x_dot = A @ x + B[:, 0] * delta1 + np.array([0.0, 0.0, 0.0, -Vx * kappa]) + w
    # x_next = x + dt * x_dot
    x_dot = A_p @ x_true + B_p[:, 0] * delta1 + E_p[:, 0] * kappa + w
    x_next = x_true + dt * x_dot

    vy[k+1], r[k+1], e_dL[k+1], e_phiL[k+1] = x_next

    # store
    s_hist[k] = s
    delta1_hist[k] = delta1
    delta2_hist[k] = delta2
    delta_ref_hist[k] = delta_ref
    delta_eq_hist[k] = delta_ref + u_eq
    delta_st_hist[k] = u_st
    r_ref_hist[k] = x_ref[1]
    ephi_ref_hist[k] = x_ref[3]
    x_ref_hist[k, :] = x_ref
    z_hist[k, :] = z
    r_meas_hist[k] = r_meas

    x_ref_prev = x_ref.copy()

# final sample
kappa_last = road_curvature(t[-1])
x_last = np.array([vy[-1], r[-1], e_dL[-1], e_phiL[-1]], dtype=float)
x_ref_last, delta_ref_last = steady_state_reference(kappa_last)
z_last = x_last - x_ref_last

s_hist[-1] = float((C @ z_last).item())
delta1_hist[-1] = delta1_hist[-2]
delta2_hist[-1] = delta2_hist[-2]
delta_ref_hist[-1] = delta_ref_last
delta_eq_hist[-1] = delta_eq_hist[-2]
delta_st_hist[-1] = delta_st_hist[-2]
r_ref_hist[-1] = x_ref_last[1]
ephi_ref_hist[-1] = x_ref_last[3]
x_ref_hist[-1, :] = x_ref_last
z_hist[-1, :] = z_last

print("Final x          =", x_last)
print("Final x_ref      =", x_ref_last)
print("Final z = x-xref =", z_last)
print("Final e_dL       =", e_dL[-1])
print("Final r          =", r[-1])
print("Final r_ref      =", x_ref_last[1])
print("Final e_phiL     =", e_phiL[-1])
print("Final e_phiL_ref =", x_ref_last[3])
print("Max |delta1|     =", np.max(np.abs(delta1_hist)))
print("C =", C)
print("Gamma =", Gamma)
print("Final s =", s_hist[-1])

# =====================
# Plots
# =====================

# 1) Main states and steering
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Truck Lateral Dynamics and Steering Response", fontsize=14)

# Lateral offset
axs[0, 0].plot(t, e_dL, label="e_dL")
axs[0, 0].plot(t, x_ref_hist[:, 2], "--", label="e_dL_ref")
axs[0, 0].set_title("Lateral Offset")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Lateral offset [m]")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Yaw rate
axs[0, 1].plot(t, r, label="r")
axs[0, 1].plot(t, r_ref_hist, "--", label="r_ref")
axs[0, 1].set_title("Yaw Rate")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Yaw rate [rad/s]")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Heading error
axs[1, 0].plot(t, e_phiL, label="e_phiL")
axs[1, 0].plot(t, ephi_ref_hist, "--", label="e_phiL_ref")
axs[1, 0].set_title("Heading Error")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Heading angle [rad]")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Steering inputs
axs[1, 1].plot(t, delta1_hist, label="delta1")
axs[1, 1].plot(t, delta2_hist, "--", label="delta2")
axs[1, 1].plot(t, delta_ref_hist, ":", label="delta_ref")
axs[1, 1].set_title("Steering Inputs")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Steering angle [rad]")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 2) Tracking errors z = x - x_ref
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Tracking Errors", fontsize=14)

error_titles = [
    "Lateral Velocity Error",
    "Yaw Rate Error",
    "Lateral Offset Error",
    "Heading Error"
]

error_labels = [
    "vy_tilde [m/s]",
    "r_tilde [rad/s]",
    "e_dL_tilde [m]",
    "e_phiL_tilde [rad]"
]

for i, ax in enumerate(axs.flat):
    ax.plot(t, z_hist[:, i])
    ax.set_title(error_titles[i])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(error_labels[i])
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 3) Sliding variable
plt.figure(figsize=(3, 3))  # smaller figure
plt.plot(t, s_hist, linewidth=2.5)  # thicker line
plt.plot(t, np.zeros_like(t), "--", label="s_ref", linewidth=2)
plt.title("Sliding Variable", fontsize=16, fontweight="bold")
plt.xlabel("Time [s]", fontsize=12, fontweight="bold")
plt.ylabel("s = C(x - x_ref)", fontsize=18, fontweight="bold")
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()



# 4) Steering control components
# Removed delta1 here because it is already shown in "Steering Inputs"
plt.figure(figsize=(10, 4))
plt.plot(t, delta_eq_hist, label="delta_eq")
plt.plot(t, delta_st_hist, label="delta_st")
plt.title("Steering Control Components")
plt.xlabel("Time [s]")
plt.ylabel("Steering angle [rad]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()