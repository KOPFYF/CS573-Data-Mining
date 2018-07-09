import math
import random
import matplotlib.pyplot as plt


# upperBound: int, int -> float
# the size of the upper confidence bound for ucb1
def upperBound(step, numPlays):
   return math.sqrt(2 * math.log(step + 1) / numPlays)


# ucb1: int, (int, int -> float) -> generator
# perform the ucb1 bandit learning algorithm.  numActions is the number of
# actions, indexed from 0. reward is a function (or callable) accepting as
# input the action and producing as output the reward for that action
def ucb1(numActions, reward):
   payoffSums = [0] * numActions
   numPlays = [1] * numActions
   ucbs = [0] * numActions

   # initialize empirical sums
   for t in range(numActions):
      payoffSums[t] = reward(t,t)
      yield t, payoffSums[t], ucbs

   t = numActions

   while True:
      ucbs = [payoffSums[i] / numPlays[i] + upperBound(t, numPlays[i]) for i in range(numActions)]
      action = max(range(numActions), key=lambda i: ucbs[i])
      theReward = reward(action, t)
      numPlays[action] += 1
      payoffSums[action] += theReward

      yield action, theReward, ucbs
      t = t + 1


# Test UCB1 using stochastic payoffs for 10 actions.
def simpleTest():
   numActions = 2
   numRounds = 10000

   biases = [1608298/(4875504+1608298)*0.1, 100815/(500908+100815)*0.1]  # case 1
   # biases = [1608298/(4875504+1608298)*0.1, 3129179/(9701453+3129179)*0.1] # case 2

   means = [0.0248, 0.0168]

   bestAction = 0
   rewards = lambda choice, t: biases[choice]

   cumulativeReward = 0
   bestActionCumulativeReward = 0

   t = numActions
   arm0,arm1 = 0,0
   arm0_reward, arm1_reward = 0,0
   arm0_cum_reward_list, arm1_cum_reward_list = [],[]
   for (choice, reward, ucbs) in ucb1(numActions, rewards):
      print('choice:',choice)
      if choice == 0:
         arm0 += 1
         arm0_reward += reward
         arm0_cum_reward_list.append(arm0_reward)
         arm1_cum_reward_list.append(arm1_reward)
      else:
         arm1 += 1
         arm1_reward += reward
         arm1_cum_reward_list.append(arm1_reward)
         arm0_cum_reward_list.append(arm0_reward)

      cumulativeReward += reward
      bestActionCumulativeReward += reward if choice == bestAction else rewards(bestAction, t)

      print("cumulativeReward: %f\tbestActionCumulativeReward: %.2f" %(cumulativeReward,bestActionCumulativeReward))
      t += 1
      if t >= numRounds:
         break

   x = range(len(arm0_cum_reward_list))
   plt.figure()
   l1, = plt.plot(x, arm0_cum_reward_list, color='blue',label='Indiana')
   l2, = plt.plot(x, arm1_cum_reward_list, color='red',label='Illinois')

   plt.legend(loc='upper right')
   plt.xlabel('the current number of purchased ads')  
   plt.ylabel('the cumulative reward')
   plt.show()
   print(biases[0],biases[1])
   return cumulativeReward

if __name__ == "__main__":
   print(simpleTest())