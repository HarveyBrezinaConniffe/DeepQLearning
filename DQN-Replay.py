import gym
import torch
from torch import nn
from collections import deque, namedtuple
import numpy as np
import random

# Define representation of transition
Transition = namedtuple("Transition", ["state", "action", "reward", "nextState"])

# Define replay memory
bufferSize = 10000
replayMem = deque(maxlen=bufferSize)

# Define network
class DQN(nn.Module):
	def __init__(self, inputDim, outputDim):
		super(DQN, self).__init__()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(inputDim, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, outputDim),
		)

	def forward(self, x):
		return self.linear_relu_stack(x)

# Initialize DQN and environment
env = gym.make('MountainCar-v0')
Qtarget = DQN(2, 3)
Qpolicy = DQN(2, 3)

batchSize = 64

def renderGame():
	observation = env.reset()
	done = False
	while not done:
		env.render()
		currentState = torch.from_numpy(observation)
		predictedRewards = Qtarget(currentState)
		action = np.argmax(predictedRewards.detach().numpy())
		observation, _, done, info = env.step(action)
	env.render()

def collectTransitions(numTransitions):
	observation = env.reset()
	avgReward = 0
	for i in range(numTransitions):
		# Get current state and convert it to numpy
		currentState = torch.from_numpy(observation)
		# Predict sum of future rewards resulting from each action
		predictedRewards = Qtarget(currentState)
		# Choose the best action and take it
		action = np.argmax(predictedRewards.detach().numpy())
		observation, _, done, info = env.step(action)
		reward = np.exp(observation[0]*10, dtype=np.float32)
		avgReward += reward
		nextState = torch.from_numpy(observation)
		transition = Transition(currentState, action, reward, nextState)	
		replayMem.append(transition)
		if done:
			observation = env.reset()
	return avgReward/numTransitions

def trainStep(lossFunc, optimizer):
	# Sample some of the transitions from the replay memory
	batch = random.sample(replayMem, batchSize)
	# Extract the states, action, rewards and nextStates from each transition in the sample
	states, actions, rewards, nextStates = zip(*batch)
	states = torch.stack(states)
	rewards = torch.from_numpy(np.array(rewards))
	nextStates = torch.stack(nextStates)
	# Predict the total future reward for each action for each state in the sample
	predictedRewards = Qtarget(states)
	# We want to get the predicted reward for the action taken in each state.
	# We will do this by building a mask and summing along the 1st dimension
	mask = torch.zeros_like(predictedRewards)
	for i in range(batchSize):
		actionTaken = actions[i]
		mask[i][actionTaken] = 1
	predictedRewards *= mask
	predictedRewards = torch.sum(predictedRewards, 1)
	# Find the maximum reward possible in each sample point after the action had been taken
	maxRewards = Qpolicy(nextStates)
	maxRewards = torch.max(maxRewards, 1).values
	# Add on the reward resulting from each action, This is what the Q network should have predicted
	maxRewards = (0.999*maxRewards)+rewards
	# Reset gradients
	optimizer.zero_grad()
	# Find the loss
	loss = lossFunc(predictedRewards, maxRewards)
	# Apply gradients
	loss.backward()
	for param in Qtarget.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()
	return loss

# Initialize loss func and optimizer
mse = nn.MSELoss()
optimizer = torch.optim.RMSprop(Qtarget.parameters())

trainStepsPerEpoch = 64
renderEvery = 50
updateEvery = 10
epoch = 0
while True:
	avgReward = collectTransitions(bufferSize//10)
	print("Epoch {} - Avg Reward {}".format(epoch, avgReward))
	for _ in range(trainStepsPerEpoch):
		trainStep(mse, optimizer)
	if epoch%updateEvery == 0:
		Qpolicy.load_state_dict(Qtarget.state_dict())
	if epoch%renderEvery == 0:
		renderGame()
	epoch += 1
