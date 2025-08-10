"""
Aufgabe 1: Mittleres Verschiebungsquadrat (MSD) mit Fit-Unsicherheit

Starte dieses Skript mehrmals mit unterschiedlichen
Werten von learner.D (z. B. 0.05, 0.25, 0.45)
und vergleiche die gemessene Steigung mit 2D.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from fp_classes import environment, agent

# 1) Umfeld + Agent initialisieren
env = environment()
learner = agent(env, D=0.45)          # <<< hier D variieren
learner.N_episodes = 20_000
learner.tmax_MSD  = 100

assert 0.0 <= learner.D <= 0.5, "Mit P_diffstep=2D darf D nur in [0, 0.5] liegen."

# 2) Trajektorien laufen lassen und MSD akkumulieren
t_axis   = np.arange(learner.tmax_MSD + 1)
msd_sum  = np.zeros_like(t_axis, dtype=float)

N = env.N_states
halfN = N / 2.0

for episode in range(learner.N_episodes):
    learner.x = env.starting_position
    prev_x    = learner.x

    u = 0.0                      # entwickelte Verschiebung: u(0) = 0
    traj_u = [0.0]               # speichere u(t) statt x(t)

    for t in range(1, learner.tmax_MSD + 1):
        learner.random_step()
        # Schritt auf dem Ring bestimmen
        dx = learner.x - prev_x
        # Unwrapping wegen PBC: minimalen Bildabstand verwenden
        if dx >  halfN:
            dx -= N
        elif dx < -halfN:
            dx += N
        # entwickelte Koordinate fortschreiben
        u += dx
        traj_u.append(u)

        prev_x = learner.x
    # MSD-Akkumulation: <u(t)^2> über Episoden
    msd_sum += np.square(traj_u)   # Länge = tmax_MSD + 1

# Erwartungswert
msd = msd_sum / learner.N_episodes

# 3) Linearer Fit:  msd(t) ca. 2 D t
t_fit = t_axis[1:]
y_fit = msd[1:]

slope, intercept, r_value, p_value, std_err = stats.linregress(t_fit, y_fit)
D_fit = 0.5 * slope
sigma_D = 0.5 * std_err  # Unsicherheit von D_fit

print(f"Gewähltes  D0   = {learner.D:.4f}")
print(f"Gemessenes Dfit = {D_fit:.4f} ± {sigma_D:.4f}  (Steigung = {slope:.4f} ± {std_err:.4f})")

# 4) Plot + Datei speichern
fig, ax = plt.subplots()
ax.plot(t_axis, msd, label=r'$\langle (x-x_0)^2\rangle$')

# Gefittete Gerade
fit_line = intercept + slope * t_axis
ax.plot(t_axis, fit_line, '--',
        label=rf'Fit: $2Dt$, $D_{{\rm fit}}={D_fit:.3f} \pm {sigma_D:.3f}$')

# Fehlerband für den Fit (nur statistische Fit-Unsicherheit)
ax.fill_between(t_axis,
                fit_line - std_err * t_axis,
                fit_line + std_err * t_axis,
                color='orange', alpha=0.3, label='Fit-Unsicherheit')

ax.set_xlabel('t')
ax.set_ylabel(r'$\langle (x-x_0)^2\rangle$')
ax.legend()
ax.set_title(rf'MSD auf dem Ring (unwrapped), $D_0={learner.D:.3f}$')

now  = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
NAME = "Chatt_Weingart"
fig.savefig(f"Aufgabe_1_MSD_D{learner.D:.2f}_{D_fit:.3f}pm{sigma_D:.3f}_{NAME}_{now}.png",
            bbox_inches="tight")
plt.show()