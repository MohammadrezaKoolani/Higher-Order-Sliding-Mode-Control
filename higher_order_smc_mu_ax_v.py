import numpy as np
import matplotlib.pyplot as plt

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

# dual-front-steering relation from the truck paper
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
    [0.0, 1.0, 0.0, 0.0]
], dtype=float)

B = np.array([
    [b1],
    [b2],
    [0.0],
    [0.0]
], dtype=float)

# curvature input enters e_phiL_dot = r - kappa_r * Vx
E = np.array([
    [0.0],
    [0.0],
    [0.0],
    [-Vx]
], dtype=float)

# =====================
# States
# =====================
vy = np.zeros(N)
r = np.zeros(N)
e_dL = np.zeros(N)
e_phiL = np.zeros(N)

e_dL[0] = 1.0
e_phiL[0] = 0.05

# ============================================
# Higher-order SMC (super-twisting) on s = Cx
# ============================================
# This C is a designed sliding surface for THIS truck model.
C = np.array([[-3.67746041, 14.76200588, 7.88453846, 11.85602271]], dtype=float)

Gamma = float((C @ B).item())

alpha = 0.45
beta = 0.01
u2 = 0.0

delta_max = np.deg2rad(60.0)

# histories
s_hist = np.zeros(N)
delta1_hist = np.zeros(N)
delta2_hist = np.zeros(N)
delta_eq_hist = np.zeros(N)
delta_st_hist = np.zeros(N)

def road_curvature(time):
    return 0.0   # straight road

for k in range(N - 1):
    kappa = road_curvature(t[k])

    x = np.array([vy[k], r[k], e_dL[k], e_phiL[k]], dtype=float)

    # sliding variable
    s = float((C @ x).item())

    # s_dot = Phi + Gamma * delta1
    Phi = float((C @ (A @ x + E[:, 0] * kappa)).item())

    # equivalent control
    delta_eq = -Phi / Gamma

    # super-twisting term
    u1 = -alpha * np.sqrt(abs(s) + 1e-12) * np.sign(s)
    u2 += dt * (-beta * np.sign(s))
    delta_st = (u1 + u2) / Gamma

    # total steering command for axle 1
    delta1 = np.clip(delta_eq + delta_st, -delta_max, delta_max)

    # axle 2 steering
    delta2 = a_s * delta1

    # truck linear model
    x_dot = A @ x + B[:, 0] * delta1 + E[:, 0] * kappa
    x_next = x + dt * x_dot

    vy[k+1], r[k+1], e_dL[k+1], e_phiL[k+1] = x_next

    # store
    s_hist[k] = s
    delta1_hist[k] = delta1
    delta2_hist[k] = delta2
    delta_eq_hist[k] = delta_eq
    delta_st_hist[k] = delta_st

# final samples
x_last = np.array([vy[-1], r[-1], e_dL[-1], e_phiL[-1]], dtype=float)
s_hist[-1] = float((C @ x_last).item())
delta1_hist[-1] = delta1_hist[-2]
delta2_hist[-1] = delta2_hist[-2]
delta_eq_hist[-1] = delta_eq_hist[-2]
delta_st_hist[-1] = delta_st_hist[-2]

# =====================
# Plots
# =====================
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(t, e_dL)
plt.xlabel("Time [s]")
plt.ylabel("e_dL [m]")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, e_phiL)
plt.xlabel("Time [s]")
plt.ylabel("e_phiL [rad]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, delta1_hist, label="delta1")
plt.plot(t, delta2_hist, "--", label="delta2")
plt.xlabel("Time [s]")
plt.ylabel("Steering [rad]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, vy, label="vy")
plt.plot(t, r, label="r")
plt.xlabel("Time [s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, s_hist)
plt.xlabel("Time [s]")
plt.ylabel("Sliding variable s")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, delta_eq_hist, label="delta_eq")
plt.plot(t, delta_st_hist, label="delta_st")
plt.plot(t, delta1_hist, label="delta1")
plt.xlabel("Time [s]")
plt.legend()
plt.grid(True)
plt.show()




