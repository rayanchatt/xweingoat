import numpy as np

class environment:
	def __init__(self):
		self.N_states = 100
		self.target_position = 8
		self.starting_position = 30
		
		self.obstacle_interval = np.arange(9,12)
		self.P_obstacle = 0.0
	
class agent:
	def __init__(self,env_):
		self.N_episodes = None
		self.tmax_MSD = None
		
		self.x = 1
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = None
		self.gamma = None
		self.epsilon = 1.0
		self.target_reward = 10.0
		self.zero_fraction = 0.9

		self.D = 0.05
		self.P_diffstep = None
		
		self.x_old = None
		
		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self):
		pass
		
	def adjust_epsilon(self,episode):
		pass
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
		"""
		pass
	def perform_action(self,env_):
		"""
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

		pass
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		pass
	
	def stoch_obstacle(self,env_):
		pass

