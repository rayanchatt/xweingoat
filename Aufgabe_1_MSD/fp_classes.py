import numpy as np


class environment:
    """Sehr einfaches 1-D-Umfeld – reicht für das MSD-Experiment."""
    def __init__(self):
        self.N_states          = 100          # Länge der x-Achse
        self.target_position   = 8            # hier ohne Bedeutung
        self.starting_position = 30           # Start-x für jede Episode

        # Hindernis-Parameter (für spätere Aufgaben)
        self.obstacle_interval = np.arange(9, 12)
        self.P_obstacle        = 0.0


class agent:
    """
    Teilchen, das sich rein diffus bewegt.
    """
    def __init__(self, env_: environment, D: float = 0.25):
        # -----------------  MSD-relevante Attribute  -----------------
        self.D           = D                         # vorgegebene Diffusions­konstante
        self.P_diffstep  = 2 * self.D                # Schritt-Wahrscheinlichkeit (a = τ = 1)
        self.x           = env_.starting_position    # aktuelle Position
        self.traj: list[int] = []                    # speichert x(t) einer Episode
        # -------------------------------------------------------------
        self.N_episodes  = None                      # wird im Hauptskript gesetzt
        self.tmax_MSD    = None

    # ----------  Diffusionsschritt  ---------------------------------
    def random_step(self) -> None:
        """Mit Wahrscheinlichkeit P_diffstep einen Schritt ±1; sonst bleibt x gleich."""
        if np.random.rand() < self.P_diffstep:
            self.x += np.random.choice((-1, 1))
        # Position in Trajektorie ablegen (wichtig für MSD-Berechnung)
        self.traj.append(self.x)

    # ----------  Platzhalter für spätere RL-Aufgaben  ---------------
    def choose_action(self, env_: environment):
        pass

    def exec_action(self, env_: environment):
        pass

    def update_Q(self, env_: environment):
        pass

    def stoch_obstacle(self, env_: environment):
        pass
