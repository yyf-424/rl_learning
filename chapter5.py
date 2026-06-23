import gridworld
import numpy as np

base_setting = {"env_size" : (5,5),
             "start_state" : (1,2),
             "target_state" : (2,3),#3.4
             "forbidden_states" :[(1, 1), (2, 1), (2,2),(1,3),(3,3),(1,4)],
             "reward_target" : 1.0,
             "reward_forbidden" : -10.0,
             "reward_boundary" : -1.0,
             "reward_step" : -0.01}

advance_setting = {"action_space" : [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)],#下，右，上，左，不动
                   "animation_interval" : 0.2,
                   "debug" : False}

env = gridworld.GridWorld(base_setting ,advance_setting )
state, info = env.reset() # 初始化绘图
env.render()

policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis] #初始化策略矩阵

values = np.random.uniform(0,10,(env.num_states,)) #初始化状态值矩阵

policy_matrix = env.mc_basic(t_episode=20000, t_iteration=10000,gamma = 0.9) 

env.add_policy(policy_matrix)

env.render(animation_interval=2)
env.show()