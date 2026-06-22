import gridworld
import numpy as np

base_setting = {"env_size" : (2,2),
             "start_state" : (0,0),
             "target_state" : (1,1),#3.4
             "forbidden_states" : [(1, 0)],
             "reward_target" : 1.0,
             "reward_forbidden" : -1.0,
             "reward_step" : 0.0}

advance_setting = {"action_space" : [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)],#下，右，上，左，不动
                   "animation_interval" : 0.2,
                   "debug" : False}

env = gridworld.GridWorld(base_setting ,advance_setting )
state, info = env.reset() # 初始化绘图
env.render()


policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis] #初始化策略矩阵

values = np.random.uniform(0,10,(env.num_states,)) #初始化状态值矩阵

p_s = env.get_p_s()
policy_matrix = env.value_iteration(p_s)

env.add_policy(policy_matrix)

env.render(animation_interval=2)
env.show()