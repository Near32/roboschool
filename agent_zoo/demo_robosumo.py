from OpenGL import GL 
import roboschool
import gym

env = gym.make('RoboschoolSumo-v0')
env.reset()
while True:
	# The first sumo does not move, while the second moves randomly:
	action = env.action_space.sample()
	action[0:8] = 0.0
	state, reward, done, info = env.step(action)
	#env.render()

	if done :
		env.reset()