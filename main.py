import gridworld 
import random
import numpy as np

base_setting = {"env_size" : (5,5),
             "start_state" : (1,2),
             "target_state" : (2,3),#3.4
             "forbidden_states" : [(1, 1), (2, 1), (2,2),(1,3),(3,3),(1,4)],
             "reward_target" : 1.0,
             "reward_forbidden" : -1.0,
             "reward_step" : 0.0}

advance_setting = {"action_space" : [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)],#下，右，上，左，不动
                   "animation_interval" : 0.2,
                   "debug" : False}



if __name__ == "__main__":
    env = gridworld.GridWorld(base_setting, advance_setting)
    state = env.reset()

    policy_matrix=np.random.rand(env.num_states,len(env.action_space))    
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  #初始化策略矩阵

    values = np.random.uniform(0,10,(env.num_states,)) #初始化状态值矩阵

    for i in range(0,100):
        env.render()
        action = random.choice(advance_setting["action_space"])
        next_state, reward, done, info = env.step(action)
        #print(f"Step: {i}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        #if done:
        #    print("!!!!!")

    env.add_policy(policy_matrix)

    # Add state values
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2)
    env.show()



        
    
