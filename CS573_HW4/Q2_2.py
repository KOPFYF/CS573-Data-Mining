import math
import random
import matplotlib.pyplot as plt
import numpy as np

def upperBound(step, numPlays):
   return math.sqrt(2 * math.log(step + 1) / numPlays)

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

   # IN_list=[0]*((4875504+1608298)*100) # 0 stands for adult
   # Ca_list=[0]*((500908+100815)*100)   # 0 stands for adult
   # IL_list=[0]*((3129179+9701453)*100)   # 0 stands for adult
   # for i in range(1608298):
   #    IN_list[i]=1 # 1 stands for pop

   # for i in range(100815):
   #    Ca_list[i]=1 # 1 stands for pop

   # for i in range(3129179):
   #    IL_list[i]=1 # 1 stands for pop

   bestAction = 0
   p1 = (1608298/(4875504+1608298))*0.3
   p2 = (100815/(500908+100815))*0.3
   p3 = (3129179/(3129179+9701453))*0.3

   # rewards = lambda choice, t: [random.sample(IN_list,1).count(1)*0.01*10,
   # random.sample(IL_list,1).count(1)*0.01*10][choice]

   rewards = lambda choice, t: [np.random.binomial(1,p1)*30-0.01, np.random.binomial(1,p2)*30-0.01][choice]

   cumulativeReward = 0
   bestActionCumulativeReward = 0

   t = numActions
   arm0,arm1 = 0,0
   arm0_reward, arm1_reward = 0,0
   arm0_cum_reward_list, arm1_cum_reward_list = [],[]
   for (choice, reward, ucbs) in ucb1(numActions, rewards):
      # biases = [IN_sample.count(1)*0.01*10, Ca_sample.count(1)*0.01*10]
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
   # print(arm0,arm1)
   # print(arm1_cum_reward_list)
   # print(arm0_cum_reward_list)
   return cumulativeReward

if __name__ == "__main__":
   print(simpleTest())