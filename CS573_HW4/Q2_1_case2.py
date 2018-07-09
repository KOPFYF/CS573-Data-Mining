import random
# Case 1
IN_list=[0]*(4875504+1608298) # 0 stands for adult
Ca_list=[0]*(3129179+9701453)   # 0 stands for adult

for i in range(1608298):
    IN_list[i]=1 # 1 stands for pop

for i in range(3129179):
    Ca_list[i]=1 # 1 stands for pop

reward_List=[] 
test_num = 10

for i in range(test_num):
    IN_sample = random.sample(IN_list,9500)
    Indi_test  = IN_sample[:500]
    Indi_9000 = IN_sample[501:9000]
    rew_indi = Indi_test.count(1)
    
    Ca_sample = random.sample(Ca_list,9500)
    Cali_test  = Ca_sample[:500]
    Cali_9000  = Ca_sample[501:9000]
    rew_cali = Cali_test.count(1)

    count_0, count_1 = 0, 0
    cost = 10000*0.01

    if Indi_test.count(1)>Cali_test.count(1):
        print('Indiana has bigger reward on the first 500 ads')
        total_count_indi = IN_sample.count(1)
        total_count_cali = Cali_test.count(1)
        
    else:
        print('Illinois has bigger reward on the first 500 ads')
        total_count_indi = Indi_test.count(1)
        total_count_cali = Ca_sample.count(1)

    revenue = (total_count_cali + total_count_indi)*0.3*30 
    cumulative_reward = revenue - cost
    reward_List.append(cumulative_reward)
    print('total cumulative reward on test %d is: %f'%(i,cumulative_reward))
    
    