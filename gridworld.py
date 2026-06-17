import sys
import numpy as np
import contextlib
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    def __init__(self, base_setting, advance_setting):
        self.env_size = base_setting["env_size"]
        self.num_states = base_setting["env_size"][0] * \
            base_setting["env_size"][1]
        self.start_state = base_setting["start_state"]
        self.target_state = base_setting["target_state"]
        self.forbidden_states = base_setting["forbidden_states"]

        self.agent_state = self.start_state
        self.action_space = advance_setting["action_space"]
        self.reward_target = base_setting["reward_target"]
        self.reward_forbidden = base_setting["reward_forbidden"]
        self.reward_step = base_setting["reward_step"]

        self.canvas = None
        self.ax = None
        #self._render_disabled = False
        self.animation_interval = advance_setting["animation_interval"]

        self.color_forbid = (0.9290, 0.6940, 0.125)
        self.color_target = (0.3010, 0.7450, 0.9330)
        self.color_policy = (0.4660, 0.6740, 0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0, 0, 1)

    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state]
        return self.agent_state, {}

    def step(self, action):
        assert action in self.action_space, "invalid action"

        next_state, reward = self.get_next_state_and_reward(
            self.agent_state, action)
        done = self._is_done(next_state)
        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store, y_store)) + 0.2 * np.array(action))
        state_store_real = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)
        self.traj.append(state_store_real)

        return self.agent_state, reward, done, {}

    def get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))

        if action == (1, 0) and x + 1 >= self.env_size[0]:
            x = self.env_size[0] - 1
            reward = self.reward_forbidden
        elif action == (-1, 0) and x - 1 < 0:
            x = 0
            reward = self.reward_forbidden
        elif action == (0, 1) and y + 1 >= self.env_size[1]:
            y = self.env_size[1] - 1
            reward = self.reward_forbidden
        elif action == (0, -1) and y - 1 < 0:
            y = 0
            reward = self.reward_forbidden
        elif new_state in self.forbidden_states:
            x,y = state
            reward = self.reward_forbidden
        elif  new_state == self.target_state:
            x,y = new_state 
            reward = self.reward_target
        else:
            x,y = new_state
            reward = self.reward_step

        return (x,y), reward
    
    def _is_done(self, state):
        return state == self.target_state

    def render(self, animation_interval = None):

        if animation_interval == None :
            animation_interval = self.animation_interval

        if self.canvas is None:
            plt.ion()
            self.canvas, self.ax = plt.subplots()
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
            self.ax.grid(True, linestyle="-", color="gray",
                         linewidth="1", axis="both")
            self.ax.set_aspect("equal")
            self.ax.invert_yaxis()
            self.ax.xaxis.set_ticks_position("top")

            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(
                    lb,
                    -0.75,
                    str(lb + 1),
                    size=10,
                    ha="center",
                    va="center",
                    color="black",
                )
            for lb in idx_labels_y:
                self.ax.text(
                    -0.75,
                    lb,
                    str(lb + 1),
                    size=10,
                    ha="center",
                    va="center",
                    color="black",
                )
            self.ax.tick_params(
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False,
                labeltop=False,
            )

            self.target_rect = patches.Rectangle(
                (self.target_state[0] - 0.5, self.target_state[1] - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor=self.color_target,
                facecolor=self.color_target,
            )
            self.ax.add_patch(self.target_rect)

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle(
                    (forbidden_state[0] - 0.5, forbidden_state[1] - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor=self.color_forbid,
                    facecolor=self.color_forbid,
                )
                self.ax.add_patch(rect)

            (self.agent_star,) = self.ax.plot(
                [], [], marker="*", color=self.color_agent, markersize=20, linewidth=0.5
            )
            (self.traj_obj,) = self.ax.plot(
                [], [], color=self.color_trajectory, linewidth=0.5
            )

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]], [self.agent_state[1]])
        traj_x, traj_y = zip(*self.traj)
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)

    def add_policy(self, policy_matrix):                  
        for state, state_action_group in enumerate(policy_matrix):    
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability)/3*dx, dy=(0.1+action_probability)/3*dy, color=self.color_policy, width=0.001, head_width=0.05))
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))
    
    def add_state_values(self, values, precision=1):
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='black')

    def show(self):
        plt.ioff()
        plt.show()


## YYF edit
    def greedy_2_matrix(self, greedy : np.ndarray):
        policy_matrix = np.zeros((self.num_states,len(self.action_space)))
        for i,r in enumerate(greedy):
            policy_matrix[i][r] = 1
        return policy_matrix
    
    def tuple_2_index(self, state):
        x,y = state
        return self.env_size[0]*y + x
    
    def index_2_tuple(self, index):
        x = index % self.env_size[0]
        y = index // self.env_size[0]
        return (x,y)
    
    def get_next_state(self, state: tuple, action: tuple):
        x,y = state
        reward = 0

        new_state = tuple(np.array(state)+ np.array(action))

        if action == (1, 0) and x + 1 >= self.env_size[0]:
            x = self.env_size[0] - 1
            reward = self.reward_forbidden
        elif action == (-1, 0) and x - 1 < 0:
            x = 0
            reward = self.reward_forbidden
        elif action == (0, 1) and y + 1 >= self.env_size[1]:
            y = self.env_size[1] - 1
            reward = self.reward_forbidden
        elif action == (0, -1) and y - 1 < 0:
            y = 0
            reward = self.reward_forbidden
        elif new_state in self.forbidden_states:
            x,y = new_state #让进
            # x,y = state #不让进
            reward = self.reward_forbidden
        elif  new_state == self.target_state:
            x,y = new_state 
            reward = self.reward_target
        else:
            x,y = new_state
            reward = self.reward_step
        return (x,y), reward

    
    def policy_2_P_pi(self, policy:np.ndarray):

        P_pi = np.zeros((self.num_states, self.num_states))

        for state_index, action_space_set in enumerate(policy):
            for i, action_probability  in enumerate(action_space_set):
                if action_probability == 0 :
                    continue

                state_tuple = self.index_2_tuple(state_index)
                Nstate, _ = self.get_next_state(state_tuple,self.action_space[i])
                Nstate_index = self.tuple_2_index(Nstate)
                P_pi[state_index][Nstate_index] = action_probability 
        print(P_pi)
        return P_pi

    def get_reward_matix(self):
        r_pi = np.zeros((self.num_states,len(self.action_space)))

        for state_index in len(self.num_states):
            for action_index in range(len(self.action_space)):
                state_tuple = self.index_2_tuple(state_index)
                _, r = self.get_next_state(state_tuple, self.action_space[action_index])
                r_pi[state_index][action_index] = r
        return r_pi
    
    def get_policy_reward(self, policy:np.ndarray):
        r_pi = np.zeros((self.num_states,1))
        
        for state_index, action_space_set in enumerate(policy):
            max_index = np.argmax(action_space_set)
            state_tuple = self.index_2_tuple(state_index)
            _,r = self.get_next_state(state_tuple,self.action_space[max_index])
            r_pi[state_index] = r
        return r_pi


    def bellman_equ_solver(self, P_pi, r, v = None, gamma = 0.9,t = int(1e6), allo_error = 1e-6):
        if v == None :
            v = np.ones((self.num_states,1))
        
        for i in range(t):
            v_ = r+gamma*P_pi@v

            if np.linalg.norm(v_-v) < allo_error :
                break
                
            v = v_

        return v_
