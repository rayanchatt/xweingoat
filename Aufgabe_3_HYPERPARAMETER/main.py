"""
Aufgabe 5.3 – Wahl der Hyperparameter  
Sinnvolles Setting: D = 0.25, α = 0.01, γ = 0.9, N = 10 000.
"""
import numpy as np
import matplotlib.pyplot as plt
from fp_classes import environment, agent
from datetime import datetime

# Zeitstempel für Dateien
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# ------------------------------------------------------------
# 1) Umfeld + Agent initialisieren
# ------------------------------------------------------------
env = environment()
learner = agent(env, D=0.25)   # Diffusion wieder aktiv (Schritt‑Wahrsch. ≈0.5)
learner.N_episodes   = 10_000
learner.alpha        = 0.01     # sinnvolle Learning‑Rate
learner.gamma        = 0.9
learner.zero_fraction = 0.3     # Exploration in den ersten 3 000 Episoden

# ------------------------------------------------------------
# 2) Trainingsschleife
# ------------------------------------------------------------
steps_per_episode = []
ratios = []               # Verhältnis Schritte / Minimal‑schritte
Q_VALUES_OVER_TIME = []   # speichert Q-Zeile für output_state

for episode in range(learner.N_episodes):
    learner.x = np.random.randint(env.N_states)   # zufällige Startposition
    learner.adjust_epsilon(episode)
    start_x = learner.x      # für Minimaldistanz

    step_counter = 0
    while True:
        step_counter += 1
        old_x = learner.x

        learner.choose_action(env)
        # learner.random_step()            # bei D=0 ohne Effekt
        reward = learner.perform_action(env)
        learner.update_Q(old_x, reward)
        # Q-Linie des output_state protokollieren
        Q_VALUES_OVER_TIME.append(learner.Q[learner.output_state].copy())

        if reward > 0:                   # Ziel erreicht
            break
    steps_per_episode.append(step_counter)
    # ---- Minimal mögliche Anzahl Schritte (periodische Ränder) ----
    dist_right = (env.target_position - start_x) % env.N_states
    dist_left  = (start_x - env.target_position) % env.N_states
    min_steps = min(dist_left, dist_right) + 1   # +1 für Verbleiben (↓) am Ziel
    ratios.append(step_counter / min_steps)

# ------------------------------------------------------------
# 3) Lernkurve (Schritte pro Episode, log)
# ------------------------------------------------------------
plt.figure()
plt.semilogy(steps_per_episode, '.')
plt.xlabel("Episode")
plt.ylabel("Schritte bis Ziel")
plt.title("Lernkurve – rand. Start, α=0.01, γ=0.9, N=10000, D=0.25")
plt.tight_layout()
plt.savefig(f"learning_curve_steps_alpha001_D025_N10000_{now}.png")

# ------------------------------------------------------------
# 4) Verhältnis Schritte / Minimaldistanz
# ------------------------------------------------------------
plt.figure()
plt.plot(ratios, '.')
plt.xlabel("Episode")
plt.ylabel("Schritte / Minimal")
plt.title("Ratio zur minimalen Schrittzahl – α=0.01, N=10000, D=0.25")
plt.tight_layout()
plt.savefig(f"learning_curve_ratio_alpha001_D025_N10000_{now}.png")

# ------------------------------------------------------------
# 5) Q-Werte des Zustands output_state über die Zeit
# ------------------------------------------------------------
Q_arr = np.array(Q_VALUES_OVER_TIME)  # shape: (episodes, 3)
plt.figure()
plt.semilogy(Q_arr[:, 0], label="←")
plt.semilogy(Q_arr[:, 1], label="↓")
plt.semilogy(Q_arr[:, 2], label="→")
plt.xlabel("Episode")
plt.ylabel(f"Q-Werte (state {learner.output_state})")
plt.title(f"Entwicklung der Q-Werte in Zustand {learner.output_state}")
plt.legend()
plt.tight_layout()
plt.savefig(f"qvalues_over_time_D025_state30_alpha001_{now}.png")

# ------------------------------------------------------------
# 6) Demo‑Episode filmen (GIF)  –  Celluloid
# ------------------------------------------------------------
# fig2, ax2 = plt.subplots(figsize=(8, 2))
# ax2.set_ylim(-0.5, 0.5)
# ax2.set_xlim(0, env.N_states - 1)
# ax2.set_yticks([])
# ax2.set_xlabel("x‑Position (Zustand)")
# ax2.set_title("Agenten‑Trajektorie – Q‑Policy (greedy)")
# ax2.axvline(env.target_position, color="g", linestyle="--")
#
# camera = Camera(fig2)
#
# # Greedy‑Policy aufnehmen
# learner.epsilon = 0.0
# learner.x = env.starting_position
#
# for _ in range(2 * env.N_states):
#     ax2.plot(learner.x, 0, "ro")        # Agent
#     camera.snap()
#
#     learner.choose_action(env)
#     learner.perform_action(env)
#
#     if learner.x == env.target_position:
#         ax2.plot(learner.x, 0, "ro")
#         camera.snap()
#         break
#
# gif_name = f"aufgabe2_trajectory_{now}.gif"
# camera.animate().save(gif_name, writer="pillow", fps=4)
# print(f"GIF gespeichert als {gif_name}")
#
# with open(f'Q_MATRIX_{now}.txt', 'w') as f:
#     f.write(str(learner.Q))
