import numpy as np


class environment:
    def __init__(self):
        self.N_states          = 100
        self.target_position   = 8
        self.starting_position = 30

        self.obstacle_interval = np.arange(9, 12)
        self.P_obstacle        = 0.0


class agent:
    def __init__(self, env_: environment, D: float = 0.25):
        self.D           = D 
        self.P_diffstep  = 2 * self.D
        self.x           = env_.starting_position 
        self.traj: list[int] = []
        self.N_episodes  = None                     
        self.tmax_MSD    = None

    #Diffusionsschritt
    def random_step(self) -> None:
        if np.random.rand() < self.P_diffstep:
            self.x += np.random.choice((-1, 1))
        self.traj.append(self.x)

    #Platzhalter für spätere RL-Aufgaben
    def choose_action(self, env_: environment):
        """
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
		"""
        pass

    def exec_action(self, env_: environment):
        """
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

        pass

    def update_Q(self, env_: environment):
        """
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
        pass

    def stoch_obstacle(self, env_: environment):
        pass
