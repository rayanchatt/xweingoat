import numpy as np


class environment:
    """Umgebung für das Q-Learning-Experiment mit diskreten Zuständen und Aktionen.
    Hindernis‑Band 9–11 mit Verschiebung nach links."""
    def __init__(self):
        self.N_states          = 15           # Ring‑länge 15, Ziel = 12, Start = 8
        self.target_position   = 12           # Zielposition für den Agenten
        self.starting_position = 8            # Start-x für jede Episode

        # Hindernis-Parameter (für spätere Aufgaben)
        self.obstacle_interval = np.arange(9, 12)
        self.P_obstacle        = 0.0


class agent:
    """
    Agent, der mittels Q-Learning lernt, sich in der Umgebung optimal zu bewegen.
    Die Klasse implementiert grundlegende Funktionen für ε-greedy Aktionsauswahl,
    Ausführung von Aktionen und Aktualisierung der Q-Werte.
    """
    def __init__(self, env_: environment, D: float = 0.00):
        self.D           = D                         # Diffusionskonstante (nicht mehr für MSD verwendet)
        self.x           = env_.starting_position    # aktuelle Position
        self.traj: list[int] = []                    # speichert x(t) einer Episode
        self.N_episodes  = None                      # wird im Hauptskript gesetzt
        self.tmax_MSD    = None
        self.Q = np.zeros((env_.N_states, 3))  # Q(s,a) für a=-1,0,+1 → a_idx = a + 1
        self.epsilon = 0.1
        self.alpha = 0.2
        self.gamma = 0.9
        self.zero_fraction = 0.5
        self.chosen_action = None
        self.target_reward = 1.0

        # Zustand, dessen Q-Werte wir über die Zeit mitloggen
        self.output_state = 30

        # Wahrscheinlichkeit für einen zufälligen Diffusionsschritt (Aufgabe 2 – Schritt 3)
        self.P_diffstep = 2 * self.D    # a = τ = 1  ⇒  P = 2D

    def adjust_epsilon(self, episode: int) -> None:
        zero_episode = int(self.zero_fraction * self.N_episodes)
        if episode < zero_episode:
            self.epsilon = 1 - (episode / zero_episode)
        else:
            self.epsilon = 0.0

    def choose_action(self, env_: environment) -> None:
        if np.random.rand() < self.epsilon:
            # Zufallswahl (0 = links, 1 = stehen, 2 = rechts)
            self.chosen_action = np.random.randint(3)
        else:
            max_indices = np.argwhere(self.Q[self.x] == np.max(self.Q[self.x])).flatten()
            # Immer den kleinsten Index nehmen, damit bei Gleichstand Vorrang "links"
            self.chosen_action = int(np.min(max_indices))

    def perform_action(self, env_: environment) -> float:
        """Führt die gewählte Aktion aus und gibt die Belohnung zurück."""
        a = self.chosen_action - 1
        self.x = (self.x + a) % env_.N_states
        if self.x == env_.target_position and self.chosen_action == 1:
            return self.target_reward
        return 0.0

    def update_Q(self, x_old: int, reward: float) -> None:
        a_idx = self.chosen_action
        max_Q = np.max(self.Q[self.x])
        self.Q[x_old, a_idx] += self.alpha * (reward + self.gamma * max_Q - self.Q[x_old, a_idx])

    # ==============================================
    # Diffusionsschritt – reiner Zufall (← oder →)
    # ==============================================
    def random_step(self) -> None:
        """Mit Wahrscheinlichkeit P_diffstep einen Schritt ±1 (periodische Ränder)."""
        if np.random.rand() < self.P_diffstep:
            step = np.random.choice((-1, 1))
            self.x = (self.x + step) % self.Q.shape[0]   # periodisch über N_states

    def stoch_obstacle(self, env_: environment) -> None:
        """
        Stochastisches Hindernis:
        Befindet sich der Agent in env_.obstacle_interval, so wird er mit
        Wahrscheinlichkeit env_.P_obstacle um 1 nach links verschoben
        (periodische Randbedingungen).
        """
        if self.x in env_.obstacle_interval and np.random.rand() < env_.P_obstacle:
            self.x = (self.x - 1) % env_.N_states
