"""
Aufgabe 2 – Q-Learning mit ε-greedy-Strategie

Der Agent lernt mittels tabellarischem Q-Learning mit festen Parametern.
"""
import matplotlib
matplotlib.use("Agg")        # Headless backend notwendig für GIF‑Erstellung
import numpy as np
import matplotlib.pyplot as plt
from fp_classes import environment, agent
from celluloid import Camera          # Celluloid‑GIF laut Aufgabenstellung
from datetime import datetime

# Zeitstempel für Dateien
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# ------------------------------------------------------------
# 1) Umfeld + Agent initialisieren
# ------------------------------------------------------------
env = environment()
learner = agent(env, D=0.125)  # D so gewählt, dass P_diffstep ≈ 0.25
learner.N_episodes = 10_000
learner.epsilon = 0.1
learner.alpha = 0.01
learner.gamma = 0.9
learner.zero_fraction = 0.3

# ------------------------------------------------------------
# 2) Trainingsschleife
# ------------------------------------------------------------
rewards = []

for episode in range(learner.N_episodes):
    learner.adjust_epsilon(episode)
    learner.x = env.starting_position
    episode_reward = 0

    for t in range(1000):                       # längere Episode höhere Chance aufs Ziel
        old_x = learner.x

        learner.choose_action(env)              # epsilon‑greedy Aktion wählen
        learner.random_step()                   # zufälliger Diffusionsschritt
        reward = learner.perform_action(env)    # Aktion ausführen, Reward erhalten
        learner.update_Q(old_x, reward)         # Q‑Update
        episode_reward += reward

        if learner.x == env.target_position and reward > 0:
            break

    rewards.append(episode_reward)

# ------------------------------------------------------------
# 3) Ergebnisse plotten
# ------------------------------------------------------------
window = 100
rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.plot(rolling, label=f'Moving average ({window} Episoden)')
plt.xlabel("Episode")
plt.ylabel("Kumulative Belohnung")
plt.title("Lernkurve – Q-Learning")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("aufgabe2_learning_curve.png")
# plt.show()    # im Agg‑Backend nicht interaktiv

# ------------------------------------------------------------
# 4) Demo‑Episode filmen (GIF)  –  Celluloid
# ------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 2))
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlim(0, env.N_states - 1)
ax2.set_yticks([])
ax2.set_xlabel("x‑Position (Zustand)")
ax2.set_title("Agenten‑Trajektorie – Q‑Policy (greedy)")
ax2.axvline(env.target_position, color="g", linestyle="--")

camera = Camera(fig2)

# Greedy‑Policy aufnehmen
learner.epsilon = 0.0
learner.x = env.starting_position

for _ in range(2 * env.N_states):
    ax2.plot(learner.x, 0, "ro")        # Agent
    camera.snap()

    learner.choose_action(env)
    learner.perform_action(env)

    if learner.x == env.target_position:
        ax2.plot(learner.x, 0, "ro")
        camera.snap()
        break

gif_name = f"aufgabe2_trajectory_{now}.gif"
camera.animate().save(gif_name, writer="pillow", fps=4)
print(f"GIF gespeichert als {gif_name}")

with open(f'Q_MATRIX_{now}.txt', 'w') as f:
    f.write(str(learner.Q))
