"""
Aufgabe 5.4 – Stochastisches Hindernis
--------------------------------------

• Ring‑Länge         : 15
• Startposition      : 8
• Zielposition       : 12
• Hindernis‑Band     : x ∈ {9,10,11}
• P_obstacle         : Wahrscheinlichkeit, in Hindernis um 1 nach links verschoben
• Aktionen           : ← (‑1), ↓ (0), → (+1)

Ausgabe: κ = sign( Σ Δx_Aktion )   (‑1 = links bevorzugt, +1 = rechts)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fp_classes import environment, agent

# ---------------------------------------------------------------------
N_EPISODES    = 20_000     # Trainingsepisoden
ALPHA         = 0.01       # Lernrate (Schritt 9: später erneut erhöhen)
GAMMA         = 0.9
ZERO_FRACTION = 0.3        # ε linear 1→0 während der ersten 30 %
D_DIFFUSION   = 0.0        # Diffusion deaktiviert
# ---------------------------------------------------------------------
# Sweep-Bereich für Schritt 11 / 13
P_VALUES = np.linspace(0, 1, 21)    # 0.00 … 1.00 (Δ = 0.05)

# Lauf-Zeitstempel für wissenschaftliche Nachvollziehbarkeit
NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def run_training(p_obst: float, alpha: float = ALPHA) -> int:
    """
    Lerne einmal für gegebenes P_obstacle.
    Liefert κ = −1 / 0 / +1 nach Beendigung der Trainings‑ und Greedy‑phase.
    """
    env = environment()
    env.P_obstacle = p_obst            # Schritt 3

    learner = agent(env, D=D_DIFFUSION)
    learner.N_episodes    = N_EPISODES
    learner.alpha         = alpha
    learner.gamma         = GAMMA
    learner.zero_fraction = ZERO_FRACTION

    eps_cut = int(ZERO_FRACTION * N_EPISODES)
    total_action_disp = 0              # Σ Δx nach Trainingsphase

    for ep in range(N_EPISODES):
        learner.x = env.starting_position   # Schritt 6
        learner.adjust_epsilon(ep)

        while True:
            old_x = learner.x
            learner.choose_action(env)
            reward = learner.perform_action(env)
            learner.stoch_obstacle(env)     # Hindernis anwenden
            learner.update_Q(old_x, reward)

            # Schritt 8: Aktions‑Verschiebung während der greedy‑Phase mitzählen
            if learner.epsilon == 0.0:
                # Δx = chosen_action‑1   (← = –1, ↓ = 0, → = +1)
                total_action_disp += (learner.chosen_action - 1)

            if reward > 0:                  # Ziel erreicht
                break

    if total_action_disp == 0:
        return 0
    return int(np.sign(total_action_disp))


# ---------------------------------------------------------------------
# Einzeldurchlauf (Schritt 10)  – hier z. B. Erwartung P₀ ≈ 0.5
# ---------------------------------------------------------------------
P0 = 0.5
kappa_single = run_training(P0)
print(f"Einzellauf:  P_obstacle = {P0:.2f}  →  κ = {kappa_single:+d}")

# ---------------------------------------------------------------------
# Sweep um den Übergang (Schritt 11 – 13)
# ---------------------------------------------------------------------
print("\nSweep über P_obstacle:")
results = []
for p in P_VALUES:
    kappa = run_training(p)
    results.append((p, kappa))
    print(f"  {p:4.2f}  →  {kappa:+d}")

# Ergebnisse in Datei sichern
with open(f"kappa_vs_P_{NOW}.txt", "w") as f_out:
    f_out.write("# P_obstacle   kappa\n")
    for p_val, k_val in results:
        f_out.write(f"{p_val:.3f}  {k_val:+d}\n")
print(f"Ergebnis-Tabelle gespeichert: kappa_vs_P_{NOW}.txt")

p_vals, k_vals = zip(*results)
plt.step(p_vals, k_vals, where="mid")
plt.xlabel(r"$P_{\mathrm{obstacle}}$")
plt.ylabel("$\\kappa$")
plt.yticks([-1, 0, 1], ["links", "0", "rechts"])
plt.title("Vorzeichen κ als Funktion von $P_{\\text{obstacle}}$")
plt.grid(True, linestyle=":")
plt.show()

# ---------------------------------------------------------------------
# Schritt 14 – Extrem hohe Lernrate  (α » 1)  – Nicht‑Konvergenz demonstrieren
# ---------------------------------------------------------------------
alpha_bad = 0.999999
print(f"\nSweep mit hoher Lernrate  α = {alpha_bad}:")
results_bad = []
for p in P_VALUES:
    kappa_bad = run_training(p, alpha=alpha_bad)
    results_bad.append((p, kappa_bad))
    print(f"  {p:4.2f}  →  {kappa_bad:+d}")

# Tabelle sichern
with open(f"kappa_vs_P_alpha{alpha_bad}_{NOW}.txt", "w") as f_out:
    f_out.write("# P_obstacle   kappa   #  α = 0.999999\n")
    for p_val, k_val in results_bad:
        f_out.write(f"{p_val:.3f}  {k_val:+d}\n")
print(f"Ergebnis‑Tabelle gespeichert: kappa_vs_P_alpha{alpha_bad}_{NOW}.txt")

# ---------------------------------------------------------------------
# Vergleichs‑Plot: normale Lernrate (oben) vs. hohe Lernrate (unten)
# ---------------------------------------------------------------------
p_good, k_good = np.array(results).T
p_bad,  k_bad  = np.array(results_bad).T

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.2))
ax1.step(p_good, k_good, where="mid", label=r"$\alpha = 0.01$")
ax1.set_title(r"$\kappa$ vs. $P_{\mathrm{obstacle}}$ – normale Lernrate")
ax1.set_ylabel(r"$\kappa$")
ax1.set_yticks([-1, 0, 1], ["links", "0", "rechts"])
ax1.grid(True, ls=":")

ax2.step(p_bad, k_bad, where="mid", color="tab:red", label=r"$\alpha = 0.999999$")
ax2.set_title("$\\kappa$ vs. $P_{\\text{obstacle}}$ – zu große Lernrate")
ax2.set_xlabel(r"$P_{\mathrm{obstacle}}$")
ax2.set_ylabel(r"$\kappa$")
ax2.set_yticks([-1, 0, 1], ["links", "0", "rechts"])
ax2.grid(True, ls=":")

fig.tight_layout()
fig.savefig(f"kappa_vs_P_comparison_{NOW}.png")
plt.show()
