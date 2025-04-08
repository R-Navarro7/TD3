import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import model.TD3.td3_utils as td3_utils
import model.TD3.TD3 as TD3
import model.TD3.OurDDPG as OurDDPG
import model.TD3.DDPG as DDPG
from utils.training_params import TrainingParams
from utils.register import register_envs



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=20):
	eval_env = gym.make(env_name)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset()[0], False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	params_list = ['params/reach_td3_params.json']
	for params_fname in params_list:
		params = TrainingParams(training_params_fname=params_fname, train=True)

		file_name = f"{params.training_params.policy}_{params.env.env_name}_{params.seed}"
		print("---------------------------------------")
		print(f"Policy: {params.training_params.policy}, Env: {params.env.env_name}, Seed: {params.seed}")
		print("---------------------------------------")
		if not os.path.exists("./rslts"):
			os.makedirs("./rslts")

		if params.training_params.save_model and not os.path.exists("./saved_models"):
			os.makedirs("./saved_models")

		# Register custom environment
		register_envs(params.env.env_name)
		# Create environment
		env = gym.make(params.env.env_name)

		# Set seeds
		env.action_space.seed(params.seed)
		torch.manual_seed(params.seed)
		np.random.seed(params.seed)
		
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

		kwparams = {
			"state_dim": state_dim,
			"action_dim": action_dim,
			"max_action": max_action,
			"discount": params.training_params.discount,
			"tau": params.training_params.tau,
		}

		# Initialize policy
		if params.training_params.policy == "TD3":
			# Target policy smoothing is scaled wrt the action scale
			kwparams["policy_noise"] = params.training_params.policy_noise * max_action
			kwparams["noise_clip"] = params.training_params.noise_clip * max_action
			kwparams["policy_freq"] = params.training_params.policy_freq
			policy = TD3.TD3(**kwparams)
		elif params.training_params.policy == "OurDDPG":
			policy = OurDDPG.DDPG(**kwparams)
		elif params.training_params.policy == "DDPG":
			policy = DDPG.DDPG(**kwparams)

		if params.training_params.load_model:
			policy_file = file_name if params.training_params.load_model == "default" else params.training_params.load_model
			policy.load(f"./saved_models/{policy_file}")

		replay_buffer = td3_utils.ReplayBuffer(state_dim, action_dim, max_size=int(params.training_params.buffer_size))
		
		# Evaluate untrained policy
		evaluations = [eval_policy(policy, params.env.env_name)]

		state, done = env.reset()[0], False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		for t in range(int(params.training_params.max_timesteps)):
			
			episode_timesteps += 1

			# Select action randomly or according to policy
			if t < params.training_params.start_timesteps:
				action = env.action_space.sample()
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * params.training_params.exploration_noise, size=action_dim)
				).clip(-max_action, max_action)

			# Perform action
			next_state, reward, done, _ = env.step(action) 
			done_bool = float(done) if done else 0
			# Store data in replay buffer
			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward

			# Train agent after collecting sufficient data
			if t >= params.training_params.start_timesteps:
				policy.train(replay_buffer, params.training_params.batch_size)

			if done: 
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset environment
				state, done = env.reset()[0], False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1 

			# Evaluate episode
			if (t + 1) % params.training_params.eval_freq == 0:
				evaluations.append(eval_policy(policy, params.env.env_name))
				np.save(f"./rslts/{file_name}", evaluations)
				if params.training_params.save_model: policy.save(f"./saved_models/{file_name}")
