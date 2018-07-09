import math

# https://blog.csdn.net/tinkle181129/article/details/50486976

def ind_max(x):

    m=max(x)
    return x.index(m)

class UCB1():

    def __init__(self,counts,values):

        self.counts=counts
        self.values=values
        return 

    def initialize(self,n_arms):

        self.counts=[0 for col in range(n_arms)]
        self.values=[0.0 for col in range(n_arms)]

    def select_arm(self):

        n_arms=len(self.counts)

        for arm in range(n_arms):
            if self.counts[arm]==0:
                return arm

        ucb_values=[0.0 for arm in range(n_arms)]
        total_counts=sum(self.counts)

        for arm in range(n_arms):
            bonus=math.sqrt((2*math.log(total_counts))/float(self.counts[arm]))
            ucb_values[arm]=self.values[arm]+bonus

        return ind_max(ucb_values)

    def update(self,chosen_arm,reward):

        self.counts[chosen_arm]=self.counts[chosen_arm]+1
        n=self.counts[chosen_arm]
        value=self.values[chosen_arm]
        new_value=((n-1)/float(n))*value+(1/float(n))*reward
        self.values[chosen_arm]=new_value

        return 

# def test_algorithm(algo,arms,num_sims,horizon):

#     chosen_arms=[0.0 for i in range(num_sims*horizon)]
#     rewards=[0.0 for i in range(num_sims*horizon)]
#     cumulative_rewards=[0.0 for i in range(num_sims*horizon)]
#     sim_nums=[0.0 for i in range(num_sims*horizon)]
#     times=[0.0 for i in range(num_sims*horizon)]

#     for sim in range(num_sims):
#         sim = sim+1
#         algo.initialize(len(arms))
#         for t in range(horizon):
#             t=t+1
#             index=(sim-1)*horizon+t-1
#             sim_nums[index]=sim
#             times[index]=t
#             chosen_arm=algo.select_arm()
#             chosen_arms[index]=chosen_arm
#             reward=arms[chosen_arms[index]].draw()
#             rewards[index]=reward

#             if t==1:
#                 cumulative_rewards[index]=reward
#             else:
#                 cumulative_rewards[index]=cumulative_rewards[index-1]+reward
#             algo.update(chosen_arm,reward)

#     return [sim_nums,times,chosen_arms,rewards,cumulative_rewards]

algo = UCB1([],[])
means=[0.3299,0.2013]
n_arms=len(means)
time=1000

algo.initialize(n_arms)
for t in range(time):
    t = t+1
    chosen_arm=algo.select_arm()

    algo.update(chosen_arm,reward)

