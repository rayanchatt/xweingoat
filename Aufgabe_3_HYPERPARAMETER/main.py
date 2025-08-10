"""
Aufgabe 5.3: Wahl der Hyperparameter
Runs:
1) D=0.25, alpha=0.01
2) D=0.0,  alpha=0.01
3) D=0.0,  alpha=0.999999
"""
import numpy as np
import matplotlib.pyplot as plt
from fp_classes import environment, agent
from datetime import datetime

NAME = "Chatt_Weingart"
now  = datetime.now().strftime("%Y-%m-%dT%H %M %S")

# Experiment-Setups (D, alpha, label-suffix)
SETUPS = [
    (0.25, 0.01,     "D025_A001"),
    (0.0,  0.01,     "D000_A001"),
    (0.0,  0.999999, "D000_A0999999"),
]

N_EPISODES = 10_000
GAMMA = 0.9
ZERO_FRACTION = 0.3


def run_one_experiment(D_val, alpha_val, tag):
    # 1) Umgebung + Agent
    env = environment()
    learner = agent(env, D=D_val)
    learner.N_episodes    = N_EPISODES
    learner.alpha         = alpha_val
    learner.gamma         = GAMMA
    learner.zero_fraction = ZERO_FRACTION

    # Zustandsindex für Q-Logging robust bestimmen
    if hasattr(learner, "output_state") and 0 <= learner.output_state < env.N_states:
        STATE_TO_LOG = learner.output_state
    else:
        STATE_TO_LOG = env.N_states // 2  # Mitte des Rings als Default

    # 2) Training
    steps_per_episode = []
    ratios = []
    Q_VALUES_OVER_TIME = []

    for episode in range(learner.N_episodes):
        learner.x = np.random.randint(env.N_states)  # zufällige Startposition
        learner.adjust_epsilon(episode)
        start_x = learner.x

        step_counter = 0
        while True:
            step_counter += 1
            old_x = learner.x

            learner.choose_action(env)
            # Zufalls-Diffusionsschritt nur, wenn D > 0
            if learner.D > 0:
                learner.random_step()

            reward = learner.perform_action(env)
            learner.update_Q(old_x, reward)

            # Q-Linie des gewählten Zustands protokollieren
            Q_VALUES_OVER_TIME.append(learner.Q[STATE_TO_LOG].copy())

            if reward > 0:  # Ziel erreicht
                break

        steps_per_episode.append(step_counter)

        dist_right = (env.target_position - start_x) % env.N_states
        dist_left  = (start_x - env.target_position) % env.N_states
        min_steps = min(dist_left, dist_right) + 1
        ratios.append(step_counter / min_steps)

    # 3) Lernkurve (Schritte pro Episode, log)
    plt.figure()
    plt.semilogy(steps_per_episode, '.')
    plt.xlabel("Episode")
    plt.ylabel("Schritte bis Ziel")
    plt.title(f"Lernkurve - rand. Start, α={alpha_val}, γ={GAMMA}, N={N_EPISODES}, D={D_val}")
    plt.tight_layout()
    plt.savefig(f"Aufgabe 5.3 Lernkurve Schritte {tag} {NAME} {now}.png")
    plt.close()

    # 4) Verhältnis Schritte / Minimaldistanz
    plt.figure()
    plt.plot(ratios, '.')
    plt.xlabel("Episode")
    plt.ylabel("Schritte / Minimal")
    plt.title(f"Ratio zur minimalen Schrittzahl - α={alpha_val}, N={N_EPISODES}, D={D_val}")
    plt.tight_layout()
    plt.savefig(f"Aufgabe 5.3 Lernkurve Ratio {tag} {NAME} {now}.png")
    plt.close()

    # 5) Q-Werte des Zustands STATE_TO_LOG über die Zeit
    Q_arr = np.array(Q_VALUES_OVER_TIME)
    plt.figure()
    plt.semilogy(Q_arr[:, 0], label="←")
    plt.semilogy(Q_arr[:, 1], label="↓")
    plt.semilogy(Q_arr[:, 2], label="→")
    plt.xlabel("Zeitschritt (über alle Episoden)")
    plt.ylabel(f"Q-Werte (state {STATE_TO_LOG})")
    plt.title(f"Entwicklung der Q-Werte in Zustand {STATE_TO_LOG} - α={alpha_val}, D={D_val}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Aufgabe 5.3 Q-Werte state{STATE_TO_LOG} {tag} {NAME} {now}.png")
    plt.close()

    qmat_name = f"Q MATRIX {tag} {NAME} {now}.txt"
    with open(qmat_name, 'w') as f:
        f.write(str(learner.Q))
    print(f"[OK] Q-Matrix gespeichert als {qmat_name}")

if __name__ == "__main__":
    for D_val, alpha_val, tag in SETUPS:
        print(f"==> Run mit D={D_val}, alpha={alpha_val} ({tag})")
        run_one_experiment(D_val, alpha_val, tag)
    print("Fertig.")