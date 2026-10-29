import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import traci
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class TrafficEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        # This is set to 'sumo' to run the simulation in the background without the GUI
        self.sumo_binary = "sumo" 
        self.sumo_config = "simulation/indian_intersection.sumocfg"
        
        self.agents = ['center']
        self._agent_ids = set(self.agents)

        self.observation_space = gym.spaces.Dict({
            "center": Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Dict({
            "center": Discrete(2) # 0 = Keep Phase, 1 = Change Phase
        })
        
        self.conn = None
        self.time = 0
        self.last_step_arrived_count = 0

    def _start_simulation(self):
        if not traci.isLoaded():
            sumo_cmd = [self.sumo_binary, "-c", self.sumo_config, "--start", "--no-warnings", "true"]
            traci.start(sumo_cmd)
        self.conn = traci

    def reset(self, *, seed=None, options=None):
        self._start_simulation()
        traci.load(["-c", self.sumo_config, "--start"])
        self.time = 0
        self.last_step_arrived_count = self.conn.simulation.getArrivedNumber()
        observations = self._get_state()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        action = action_dict['center']
        if action == 1:
            current_phase = self.conn.trafficlight.getPhase('center')
            next_phase = (current_phase + 2) % 4 # Skip yellow phases
            self.conn.trafficlight.setPhase('center', next_phase)

        for _ in range(10):
            self.conn.simulationStep()
            self.time += 1
        
        observations = self._get_state()
        
        total_halting_cars = observations['center'].sum()
        wait_penalty = -total_halting_cars
        
        current_arrived_count = self.conn.simulation.getArrivedNumber()
        throughput_reward = current_arrived_count - self.last_step_arrived_count
        self.last_step_arrived_count = current_arrived_count
        
        reward = wait_penalty + (throughput_reward * 2.5)
        
        rewards = {'center': reward}
        infos = {'center': {}}
        
        terminateds = {"__all__": False}
        truncateds = {"__all__": self.time > 3600}

        return observations, rewards, terminateds, truncateds, infos

    def _get_state(self):
        lanes = self.conn.trafficlight.getControlledLanes('center')
        state = [self.conn.lane.getLastStepHaltingNumber(lane) for lane in lanes]
        
        padded_state = np.zeros(4, dtype=np.float32)
        padded_state[:len(state)] = state
        return {'center': padded_state}

    def close(self):
        if self.conn and traci.isLoaded():
            traci.close()
            self.conn = None