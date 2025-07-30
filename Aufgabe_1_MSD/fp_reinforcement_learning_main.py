"""
Aufgabe 1 – Mittleres Verschiebungsquadrat  (MSD)

Starte dieses Skript mehrmals mit unterschiedlichen
Werten von learner.D (z. B. 0.05, 0.25, 0.45)
und vergleiche die gemessene Steigung mit 2 D.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # nur für den linearen Fit

from fp_classes import environment, agent

# ------------------------------------------------------------
# 1) Umfeld + Agent initialisieren
# ------------------------------------------------------------
env = environment()
learner = agent(env, D=0.25)           # <<<  hier D variieren
learner.N_episodes = 20_000
learner.tmax_MSD  = 100

# ------------------------------------------------------------
# 2) Trajektorien laufen lassen und MSD akkumulieren
# ------------------------------------------------------------
t_axis      = np.arange(learner.tmax_MSD + 1)
msd_running = np.zeros_like(t_axis, dtype=float)

for episode in range(learner.N_episodes):
    learner.x    = env.starting_position
    learner.traj = [learner.x]         # x(0)
    for t in range(1, learner.tmax_MSD + 1):
        learner.random_step()
    # Trajektorie in msd_running einsammeln
    msd_running += np.square(learner.traj + [learner.x][-1])  # len = tmax+1

# Erwartungswert ⟨x²⟩
msd = msd_running / learner.N_episodes

# ------------------------------------------------------------
# 3) Linearer Fit   msd(t) ≈ 2 D t  (+ Offset)
# ------------------------------------------------------------
slope, intercept, *_ = stats.linregress(t_axis[1:], msd[1:])
D_fit = slope / 2
print(f"Gewähltes   D0 = {learner.D:.4f}")
print(f"Gemessenes Dfit = {D_fit:.4f}  (Steigung = {slope:.4f})")

# ------------------------------------------------------------
# 4) Plot + Datei speichern
# ------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(t_axis, msd, label=r'$\langle x^{2}(t)\rangle$')
ax.plot(t_axis, intercept + slope * t_axis,
        '--', label=rf'Fit: $2Dt$,  $D_\mathrm{{fit}}={D_fit:.3f}$')
ax.set_xlabel('t')
ax.set_ylabel(r'$\langle x^{2}\rangle$')
ax.legend()

now = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
fig.savefig(f'Aufgabe1_MSD_D{learner.D:.2f}_{now}.png', bbox_inches="tight")
plt.show()
