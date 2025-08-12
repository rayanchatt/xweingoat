"""
Aufgabe 5.4 - Stochastisches Hindernis
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fp_classes import environment, agent

# Params
N_EPISODES    = 20_000
ALPHA         = 0.01
GAMMA         = 0.9
ZERO_FRACTION = 0.3
D_DIFFUSION   = 0.0

# Sweep-Bereich
P_VALUES = np.linspace(0, 1, 21)

NAME = "Chatt_Weingart"
NOW  = datetime.now().strftime("%Y-%m-%dT%H %M %S")

def run_training(p_obst: float, alpha: float = ALPHA) -> int:
    """Training mit gegebenem P_obstacle. Liefert κ = -1 / 0 / +1."""
    env = environment()
    env.P_obstacle = p_obst

    learner = agent(env, D=D_DIFFUSION)
    learner.N_episodes    = N_EPISODES
    learner.alpha         = alpha
    learner.gamma         = GAMMA
    learner.zero_fraction = ZERO_FRACTION

    total_action_disp = 0

    for ep in range(N_EPISODES):
        learner.x = env.starting_position
        learner.adjust_epsilon(ep)

        while True:
            old_x = learner.x
            learner.choose_action(env)
            reward = learner.perform_action(env)
            learner.stoch_obstacle(env)
            learner.update_Q(old_x, reward)

            if learner.epsilon == 0.0:
                total_action_disp += (learner.chosen_action - 1)

            if reward > 0:
                break

    if total_action_disp == 0:
        return 0
    return int(np.sign(total_action_disp))

# Einzeldurchlauf
P0 = 0.5
kappa_single = run_training(P0)
print(f"Einzellauf:  P_obstacle = {P0:.2f}  →  κ = {kappa_single:+d}")

# Sweep normale Lernrate
print("\nSweep über P_obstacle:")
results = []
for p in P_VALUES:
    kappa = run_training(p)
    results.append((p, kappa))
    print(f"  {p:4.2f}  →  {kappa:+d}")

table_good = f"kappa_vs_P_{NAME}_{NOW}.txt"
with open(table_good, "w") as f_out:
    f_out.write("# P_obstacle   kappa\n")
    for p_val, k_val in results:
        f_out.write(f"{p_val:.3f}  {k_val:+d}\n")
print(f"[OK] Ergebnis-Tabelle gespeichert: {table_good}")

# Plot normale Lernrate
p_vals, k_vals = zip(*results)
plt.step(p_vals, k_vals, where="mid")
plt.xlabel(r"$P_{\mathrm{obstacle}}$")
plt.ylabel(r"$\kappa$")
plt.yticks([-1, 0, 1], ["links", "0", "rechts"])
plt.title(r"Vorzeichen $\kappa$ als Funktion von $P_{\text{obstacle}}$")
plt.grid(True, linestyle=":")
plt.tight_layout()
plot_good = f"kappa_vs_P_{NAME}_{NOW}.png"
plt.savefig(plot_good)
print(f"[OK] Plot gespeichert: {plot_good}")
plt.close()

# Sweep hohe Lernrate
alpha_bad = 0.999999
print(f"\nSweep mit hoher Lernrate  α = {alpha_bad}:")
results_bad = []
for p in P_VALUES:
    kappa_bad = run_training(p, alpha=alpha_bad)
    results_bad.append((p, kappa_bad))
    print(f"  {p:4.2f}  →  {kappa_bad:+d}")

table_bad = f"kappa_vs_P_alpha{alpha_bad}_{NAME}_{NOW}.txt"
with open(table_bad, "w") as f_out:
    f_out.write("# P_obstacle   kappa   #  α = 0.999999\n")
    for p_val, k_val in results_bad:
        f_out.write(f"{p_val:.3f}  {k_val:+d}\n")
print(f"[OK] Ergebnis-Tabelle gespeichert: {table_bad}")


# Vergleichsplot
p_good, k_good = np.array(results).T
p_bad,  k_bad  = np.array(results_bad).T

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.2))
ax1.step(p_good, k_good, where="mid", label=r"$\alpha = 0.01$")
ax1.set_title(r"$\kappa$ vs. $P_{\mathrm{obstacle}}$ – normale Lernrate")
ax1.set_ylabel(r"$\kappa$")
ax1.set_yticks([-1, 0, 1], ["links", "0", "rechts"])
ax1.grid(True, ls=":")

ax2.step(p_bad, k_bad, where="mid", color="tab:red", label=r"$\alpha = 0.999999$")
ax2.set_title(r"$\kappa$ vs. $P_{\text{obstacle}}$ – zu große Lernrate")
ax2.set_xlabel(r"$P_{\mathrm{obstacle}}$")
ax2.set_ylabel(r"$\kappa$")
ax2.set_yticks([-1, 0, 1], ["links", "0", "rechts"])
ax2.grid(True, ls=":")

fig.tight_layout()
plot_comp = f"kappa_vs_P_comparison_{NAME}_{NOW}.png"
fig.savefig(plot_comp)
print(f"[OK] Vergleichsplot gespeichert: {plot_comp}")
plt.close()