import numpy as np
import matplotlib.pyplot as plt

T = 60.0
dt = 0.01
N = int(T / dt) + 1
t = np.arange(0, T + dt, dt)

# States
e = np.zeros(N)       # lateral tracking error
edot = np.zeros(N)    # derivative of lateral tracking error
vy = np.zeros(N)      # lateral velocity y_dot
r = np.zeros(N)       # yaw rate psi_dot
psi = np.zeros(N)     # yaw angle

# Save control variables
s_hist = np.zeros(N)      # sliding surface
delta_hist = np.zeros(N)  # steering angle

e[0] = 1.0
u2 = 0.0

# Vehicle parameters
m = 1719
Cf = 170550
Cr = 137844
Iz = 3300
Lf = 1.195
Lr = 1.513
Vx = 13.5

# Control parameters
lam = 8.0
alpha = 0.002
beta = 0.0001

R = np.inf  # straight road

for k in range(N - 1):
    curv = 0.0 if np.isinf(R) else 1.0 / R

    # Sliding surface
    s = edot[k] + lam * e[k]
    s_hist[k] = s

    # Eq. (12)
    phi = (-(Cf + Cr) / (m * Vx)) * vy[k] \
          - ((Lf * Cf - Lr * Cr) / (m * Vx)) * r[k] \
          - Vx**2 * curv \
          + lam * edot[k]

    # Eq. (14)
    delta_eq = -(m / Cf) * phi

    # Eq. (13)
    u1 = -alpha * np.sqrt(abs(s)) * np.sign(s)
    delta_st = u1 + u2
    delta = delta_eq + delta_st
    delta_hist[k] = delta

    # explicit Euler update of u2
    u2 += dt * (-beta * np.sign(s))

    # Eq. (1): vehicle model
    vy_dot = (-(Cf + Cr) / (m * Vx)) * vy[k] \
             - (((Lf * Cf - Lr * Cr) / (m * Vx)) + Vx) * r[k] \
             + (Cf / m) * delta

    r_dot = (-(Lf * Cf - Lr * Cr) / (Iz * Vx)) * vy[k] \
            - ((Lf**2 * Cf + Lr**2 * Cr) / (Iz * Vx)) * r[k] \
            + (Lf * Cf / Iz) * delta

    # Eq. (9): tracking-error dynamics
    e_ddot = (-(Cf + Cr) / (m * Vx)) * vy[k] \
             - ((Lf * Cf - Lr * Cr) / (m * Vx)) * r[k] \
             - Vx**2 * curv \
             + (Cf / m) * delta

    vy[k+1] = vy[k] + dt * vy_dot
    r[k+1] = r[k] + dt * r_dot
    psi[k+1] = psi[k] + dt * r[k]

    edot[k+1] = edot[k] + dt * e_ddot
    e[k+1] = e[k] + dt * edot[k]

# Save final values
s_hist[-1] = edot[-1] + lam * e[-1]
delta_hist[-1] = delta_hist[-2]

# Plot 1: lateral error
plt.figure()
plt.plot(t, e)
plt.xlabel("Time (s)")
plt.ylabel("Lateral Error e (m)")
plt.grid()

# Plot 2: steering angle
plt.figure()
plt.plot(t, delta_hist)
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle δ (rad)")
plt.title("Steering Angle")
plt.grid()

# Plot 3: sliding surface
plt.figure()
plt.plot(t, s_hist)
plt.xlabel("Time (s)")
plt.ylabel("Sliding Surface s")
plt.title("Sliding Surface")
plt.grid()

plt.show()