"""
Aufgabe 2: Q-Learning mit epsilon-greedy-Strategie
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from fp_classes import environment, agent
from celluloid import Camera
from datetime import datetime

NAME = "Chatt_Weingart"
now = datetime.now().strftime("%Y-%m-%dT%H %M %S")

#Umfeld + Agent initialisieren
env = environment()
learner = agent(env, D=0.125)  # D so gewählt, dass P_diffstep ca. 0.25
learner.N_episodes = 10_000
learner.epsilon = 0.1
learner.alpha = 0.01
learner.gamma = 0.9
learner.zero_fraction = 0.3

#Trainingsschleife
rewards = []

for episode in range(learner.N_episodes):
    learner.adjust_epsilon(episode)
    learner.x = env.starting_position
    episode_reward = 0

    for t in range(1000):
        old_x = learner.x
        learner.choose_action(env)
        learner.random_step()
        reward = learner.perform_action(env)
        learner.update_Q(old_x, reward)
        episode_reward += reward

        if learner.x == env.target_position and reward > 0:
            break

    rewards.append(episode_reward)

#Ergebnisse plotten
window = 100
rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')

plt.plot(rolling, label=f'Moving average ({window} Episoden)')
plt.xlabel("Episode")
plt.ylabel("Kumulative Belohnung")
plt.title("Lernkurve – Q-Learning")
plt.legend()
plt.grid()
plt.tight_layout()

lc_name = f"Aufgabe 2 Lernkurve {NAME} {now}.png"
plt.savefig(lc_name)
print(f"Lernkurve gespeichert als {lc_name}")

#Demo-Episode
fig2, ax2 = plt.subplots(figsize=(8, 2))
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlim(0, env.N_states - 1)
ax2.set_yticks([])
ax2.set_xlabel("x-Position (Zustand)")
ax2.set_title("Agenten-Trajektorie - Q-Policy (greedy)")
ax2.axvline(env.target_position, linestyle="--")

camera = Camera(fig2)

learner.epsilon = 0.0
learner.x = env.starting_position

for _ in range(2 * env.N_states):
    ax2.plot(learner.x, 0, "o")
    camera.snap()

    learner.choose_action(env)
    learner.perform_action(env)

    if learner.x == env.target_position:
        ax2.plot(learner.x, 0, "o")
        camera.snap()
        break

gif_name = f"Aufgabe 2 Trajektorie {NAME} {now}.gif"
camera.animate().save(gif_name, writer="pillow", fps=4)
print(f"GIF gespeichert als {gif_name}")

qmat_name = f"Q MATRIX {NAME} {now}.txt"
with open(qmat_name, 'w') as f:
    f.write(str(learner.Q))
print(f"Q-Matrix gespeichert als {qmat_name}")