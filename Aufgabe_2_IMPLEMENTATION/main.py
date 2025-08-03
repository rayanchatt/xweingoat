"""
Aufgabe 2 – Q-Learning mit ε-greedy-Strategie

Der Agent lernt mittels tabellarischem Q-Learning mit festen Parametern.
"""
import numpy as np
import matplotlib.pyplot as plt
from fp_classes import environment, agent

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

    for t in range(1000):                       # längere Episode → höhere Chance aufs Ziel
        old_x = learner.x

        learner.choose_action(env)              # ε‑greedy Aktion wählen
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
plt.show()
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
with open(f'Q_MATRIX_{now}.txt', 'w') as f:
    f.write(str(learner.Q))
