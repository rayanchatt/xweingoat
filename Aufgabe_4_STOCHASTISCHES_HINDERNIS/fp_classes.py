import numpy as np


class environment:
    def __init__(self):
        self.N_states          = 15           # Ring‑länge 15, Ziel = 12, Start = 8
        self.target_position   = 12
        self.starting_position = 8

        self.obstacle_interval = np.arange(9, 12)
        self.P_obstacle        = 0.0


class agent:
    def __init__(self, env_: environment, D: float = 0.00):
        self.D           = D
        self.x           = env_.starting_position
        self.traj: list[int] = []
        self.N_episodes  = None
        self.tmax_MSD    = None
        self.Q = np.zeros((env_.N_states, 3))
        self.epsilon = 0.1
        self.alpha = 0.2
        self.gamma = 0.9
        self.zero_fraction = 0.5
        self.chosen_action = None
        self.target_reward = 1.0
        self.output_state = 30

        # Wahrscheinlichkeit für einen zufälligen Diffusionsschritt
        self.P_diffstep = 2 * self.D    # a = tau = 1, P = 2D

    def adjust_epsilon(self, episode: int) -> None:
        zero_episode = int(self.zero_fraction * self.N_episodes)
        if episode < zero_episode:
            self.epsilon = 1 - (episode / zero_episode)
        else:
            self.epsilon = 0.0

    def choose_action(self, env_: environment) -> None:
        """
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
		"""
        if np.random.rand() < self.epsilon:
            # Zufallswahl (0 = links, 1 = stehen, 2 = rechts)
            self.chosen_action = np.random.randint(3)
        else:
            max_indices = np.argwhere(self.Q[self.x] == np.max(self.Q[self.x])).flatten()
            # Immer den kleinsten Index nehmen, damit bei Gleichstand Vorrang "links"
            self.chosen_action = int(np.min(max_indices))

    def perform_action(self, env_: environment) -> float:
        """
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""
        a = self.chosen_action - 1
        self.x = (self.x + a) % env_.N_states
        if self.x == env_.target_position and self.chosen_action == 1:
            return self.target_reward
        return 0.0

    def update_Q(self, x_old: int, reward: float) -> None:
        """
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
        a_idx = self.chosen_action
        max_Q = np.max(self.Q[self.x])
        self.Q[x_old, a_idx] += self.alpha * (reward + self.gamma * max_Q - self.Q[x_old, a_idx])

    # Diffusionsschritt
    def random_step(self) -> None:
        if np.random.rand() < self.P_diffstep:
            step = np.random.choice((-1, 1))
            self.x = (self.x + step) % self.Q.shape[0]   # periodisch über N_states

    def stoch_obstacle(self, env_: environment) -> None:
        if self.x in env_.obstacle_interval and np.random.rand() < env_.P_obstacle:
            self.x = (self.x - 1) % env_.N_states
