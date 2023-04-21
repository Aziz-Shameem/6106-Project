from audioop import avg
from re import S
from turtle import pu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard import summary
from tqdm import tqdm
import operator
import argparse
import sys
import matplotlib.pyplot as plt
plt.style.use('data/plots_paper.mplstyle')
import pathlib
import warnings
warnings.filterwarnings("ignore")


class algs:
    def __init__(self, exp='genre', p=0.0, padval=0.0, true_means='true_means_test', rand='False'):

        assert(exp in ['genre', 'movie', 'book'])
        self.exp_name = exp
        self.p = p
        self.padval = padval
        self.true_means = true_means
        self.rand = rand

        #Load data
        # self.tables = pd.read_pickle(f'preproc/{self.exp_name}s/{self.exp_name}_tables_train.pkl')
        self.tables = pd.read_csv(f'preproc/{self.exp_name}s/{self.exp_name}_table.csv')
        self.test_data = pd.read_pickle(f'preproc/{self.exp_name}s/test_data_usercount')
        self.true_means_test = pd.read_pickle(f'preproc/{self.exp_name}s/{self.true_means}')

        self.numArms = len(self.true_means_test)
        self.optArm = np.argmax(self.true_means_test)
        self.numPulls = np.zeros(self.numArms)

        self.data = pd.read_pickle(f'preproc/{self.exp_name}s/data_with_id')
        
        # CODE TO FORM TABLE FROM PICKLE
        # for i in range(self.numArms) :
        #     temp = pd.DataFrame.from_dict(self.data[self.data[f'{self.exp_name}_col'] == i])
        #     if i==0 :
        #         temp.to_csv(r'C:\Users\Aziz_Shameem\OneDrive\Documents\EE6106\Project\githubRepo\correlated_bandits\preproc\genres\data_with_id.csv')
        #     else :
        #         temp.to_csv(r'C:\Users\Aziz_Shameem\OneDrive\Documents\EE6106\Project\githubRepo\correlated_bandits\preproc\genres\data_with_id.csv', mode='a')

        # print('Values loaded in data_with_id.csv...')
        
        

    def generate_samples_random(self, arm) :
        true_means = self.true_means_test
        sample = np.random.normal(true_means[arm], 1)
        return max(0, min(sample, 5))
    def generate_sample_from_data(self, arm):

        d = self.test_data[self.test_data[f'{self.exp_name}_col'] == arm]
        reward = d['Rating'].sample(n=1, replace = True)
        # print(f'reward : {reward}, type : {type(reward)}')
        # print(f'before : {type(reward)}, {reward}')

        temp = np.random.normal(self.true_means_test[arm], 0.5)
        temp = 5 if temp>5 else temp
        temp = 1 if temp<1 else temp
        reward = reward*0 + int(temp)
        # print(f'after : {type(reward)}, {reward}')

        return reward
    
    def generate_sample(self, arm) :
        if self.rand :
            return self.generate_samples_random(arm)
        else :
            return self.generate_sample_from_data(arm)

    def ThompsonSample(self, empiricalMean, numPulls, beta):
        numArms = self.numArms
        sampleArm = np.zeros(numArms)

        var_ = beta/(numPulls + 1.)
        std_ = np.sqrt(var_)

        mean_ = empiricalMean

        sampleArm = np.random.normal(mean_, std_)

        return sampleArm

    def epsilon_greedy(self, num_iterations, T) :

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables
        epsilon = 0.1 # can be changed 

        B = [5.] * numArms

        avg_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)) :
            sumReward = np.zeros(numArms)
            empReward = np.zeros(numArms)
            numPulls = np.zeros(numArms)
            regret = np.zeros(T)

            for t in range(T) :
                prob = np.random.random()
                if prob < epsilon :
                    # explore
                    pull = np.random.choice(numArms)
                else :
                    # exploit
                    pull = np.argmax(empReward)

                numPulls[pull] += 1
                reward = self.generate_sample(pull)
                sumReward[pull] += reward
                empReward[pull] = sumReward[pull] / float(numPulls[pull])

                if t==0 :
                    regret[t] = true_means_test[optArm] - true_means_test[pull]
                else :
                    regret[t] = regret[t-1] + true_means_test[optArm] - true_means_test[pull]
            avg_regret[iteration, :] = regret

        print(f'Num Pulls : EG  : {numPulls}')
        return avg_regret

    def GB(self, num_iterations, T) :

        def softmax(arr) :
            temp = np.exp(arr)
            return (temp / np.sum(temp))
        
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables
        alpha = 0.1 # can be changed

        B = [5.] * numArms
        avg_gb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)) :
            sumReward = np.zeros(numArms)
            empReward = np.zeros(numArms)
            numPulls = np.zeros(numArms)
            regret = np.zeros(T)
            prefs = np.ones(numArms)

            for t in range(T) :
                probs =  softmax(prefs)
                pull = np.random.choice(np.arange(numArms), p=probs)
                numPulls[pull] += 1
                reward = self.generate_sample(pull)
                sumReward[pull] += reward
                empReward[pull] = sumReward[pull] / float(numPulls[pull])
                totReward = np.sum(sumReward)
                avgReward = totReward / np.sum(numPulls)

                prefs = prefs - alpha*(float(reward) - avgReward)*probs
                prefs[pull] +=  alpha*(float(reward) - avgReward)

                if t==0 :
                    regret[t] = true_means_test[optArm] - true_means_test[pull]
                else :
                    regret[t] = regret[t-1] + true_means_test[optArm] - true_means_test[pull]
            avg_gb_regret[iteration, :] = regret
        print(f'Num Pulls : GB  : {numPulls}')
        return avg_gb_regret

    


    def UCB(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_ucb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            UCB_pulls = np.zeros(numArms)
            UCB_index = np.zeros(numArms)
            UCB_empReward = np.zeros(numArms)
            UCB_sumReward = np.zeros(numArms)

            UCB_index[:] = np.inf
            UCB_empReward[:] = np.inf

            ucb_regret = np.zeros(T)
            for t in range(T):
                if t < numArms:
                    UCB_kt = t
                else:
                    UCB_kt = np.argmax(UCB_index)

                UCB_reward = self.generate_sample(UCB_kt)

                UCB_pulls[UCB_kt] = UCB_pulls[UCB_kt] + 1

                UCB_sumReward[UCB_kt] = UCB_sumReward[UCB_kt] + UCB_reward
                UCB_empReward[UCB_kt] = UCB_sumReward[UCB_kt]/float(UCB_pulls[UCB_kt])

                for k in range(numArms):
                    if UCB_pulls[k] > 0:
                        UCB_index[k] = UCB_empReward[k] + B[k]*np.sqrt(2. * np.log(t+1)/ UCB_pulls[k])

                if t == 0:
                    ucb_regret[t] = true_means_test[optArm] - true_means_test[UCB_kt]
                else:
                    ucb_regret[t] = ucb_regret[t-1] + true_means_test[optArm] - true_means_test[UCB_kt]

            avg_ucb_regret[iteration, :] = ucb_regret

        print(f'Num Pulls : UCB : {UCB_pulls}')
        return avg_ucb_regret

    def TS(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        beta = 4.

        avg_ts_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            ts_regret = np.zeros(T)
            for t in range(T):
                #Initialise by pulling each arm once
                if t < numArms:
                    numPulls[t] += 1
                    assert numPulls[t] == 1

                    reward = self.generate_sample(t)
                    empReward[t] = reward

                    if t != 0:
                        ts_regret[t] = ts_regret[t-1] + true_means_test[optArm] - true_means_test[t]

                    continue

                thompson = self.ThompsonSample(empReward,numPulls,beta)
                next_arm = np.argmax(thompson)

                #Generate reward, update pulls and empirical reward
                reward = self.generate_sample( next_arm )
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                ts_regret[t] = ts_regret[t-1] + true_means_test[optArm] - true_means_test[next_arm]

            avg_ts_regret[iteration, :] = ts_regret

        print(f'Num Pulls : TS  : {numPulls}')
        return avg_ts_regret

    def C_epsilon_greedy(self, num_iterations, T) :

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables
        epsilon = 0.1 # can be changed 

        B = [5.] * numArms

        avg_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)) :
            sumReward = np.zeros(numArms)
            empReward = np.zeros(numArms)
            numPulls = np.zeros(numArms)
            regret = np.zeros(T)

            empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            sumPseudoReward = np.zeros((numArms, numArms))

            for t in range(T) :

                bool_ell = numPulls >= (float(t - 1)/numArms)

                max_mu_hat = np.max(empReward[bool_ell])

                if empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(empReward == max_mu_hat)[0][0]

                #Set of competitive arms - update through the run
                min_phi = np.min(empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                #Adding back the argmax arm
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                prob = np.random.random()
                if prob < epsilon :
                    # explore
                    if len(comp_set) == 0 :
                        pull -= np.random.choice(numArms)
                    else :
                        pull = np.random.choice(list(comp_set))
                else :
                    # exploit
                    pull = np.argmax(empReward)

                numPulls[pull] += 1
                reward = self.generate_sample(pull)
                sumReward[pull] += reward
                empReward[pull] = sumReward[pull] / float(numPulls[pull])

                # pseudoRewards = tables[pull][reward-1, :]
                pseudoRewards = np.array(tables[tables.pull==pull][tables.reward==reward.values[0]-1]).reshape(-1)[2:]
                sumPseudoReward[:, pull] = sumPseudoReward[:, pull] + pseudoRewards
                empPseudoReward[:, pull] = np.divide(sumPseudoReward[:, pull], float(numPulls[pull]))

                if t==0 :
                    regret[t] = true_means_test[optArm] - true_means_test[pull]
                else :
                    regret[t] = regret[t-1] + true_means_test[optArm] - true_means_test[pull]
            avg_regret[iteration, :] = regret
        print(f'Num Pulls : CEG : {numPulls}')
        return avg_regret

    def C_GB(self, num_iterations, T) :

        def softmax(arr) :
            temp = np.exp(arr)
            return temp / np.sum(temp)
        
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables
        alpha = 0.1 # can be changed

        B = [5.] * numArms
        avg_gb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)) :

            sumReward = np.zeros(numArms)
            empReward = np.zeros(numArms)
            pulls = np.zeros(numArms)
            numPulls = np.zeros(numArms)
            regret = np.zeros(T)
            prefs = np.ones(numArms)

            empReward[:] = np.inf

            empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            sumPseudoReward = np.zeros((numArms, numArms))

            empPseudoReward[:,:] = np.inf

            for t in range(T) :
                bool_ell = pulls >= (float(t - 1)/numArms)
                max_mu_hat = np.max(empReward[bool_ell])
                if empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(empReward == max_mu_hat)[0][0]

                min_phi = np.min(empPseudoReward[:, bool_ell], axis=1)
                comp_set = set()
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                probs =  softmax(prefs)
                if np.isnan(probs).any() :
                    print(f'probs : {probs}, prefs : {prefs}') # used for debugging
                # pull = np.random.choice(np.arange(numArms), p=probs)
                work_probs = np.array([probs[i] for i in list(comp_set)])
                if t < numArms :
                    pull = t
                else :
                    if len(comp_set) == 0 :
                        pull = np.random.choice(np.arange(numArms), p=probs)
                    else :
                        pull = np.random.choice(list(comp_set), p=work_probs/np.sum(work_probs)) 
                numPulls[pull] += 1
                pulls[pull] = pulls[pull] + 1
                reward = self.generate_sample(pull)
                sumReward[pull] += reward
                empReward[pull] = sumReward[pull] / float(numPulls[pull])

                pseudoRewards = np.array(tables[tables.pull==pull][tables.reward==reward.values[0]-1]).reshape(-1)[2:]

                sumPseudoReward[:, pull] = sumPseudoReward[:, pull] + pseudoRewards
                empPseudoReward[:, pull] = np.divide(sumPseudoReward[:, pull], float(pulls[pull]))

                #Diagonal elements of pseudorewards
                empPseudoReward[np.arange(numArms), np.arange(numArms)] = empReward


                totReward = np.sum(sumReward)
                avgReward = totReward / np.sum(numPulls)

                prefs = prefs - alpha*(float(reward) - avgReward)*probs
                prefs[pull] +=  alpha*(float(reward) - avgReward)

                if t==0 :
                    regret[t] = true_means_test[optArm] - true_means_test[pull]
                else :
                    regret[t] = regret[t-1] + true_means_test[optArm] - true_means_test[pull]
            avg_gb_regret[iteration, :] = regret

        print(f'Num Pulls : CGB : {numPulls}')
        return avg_gb_regret


    def C_UCB(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_cucb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            pulls = np.zeros(numArms)
            empReward = np.zeros(numArms)
            sumReward = np.zeros(numArms)
            Index = dict(zip(range(numArms), [np.inf]*numArms))

            empReward[:] = np.inf

            empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            sumPseudoReward = np.zeros((numArms, numArms))

            empPseudoReward[:,:] = np.inf


            cucb_regret = np.zeros(T)
            for t in range(T):

                #add to set \ell for arms with pulls >t/K
                bool_ell = pulls >= (float(t - 1)/numArms)

                max_mu_hat = np.max(empReward[bool_ell])

                if empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(empReward == max_mu_hat)[0][0]

                #Set of competitive arms - update through the run
                min_phi = np.min(empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                #Adding back the argmax arm
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                if t < numArms:
                    k_t = t %numArms
                elif len(comp_set)==0:
                    #UCB for empty comp set
                    k_t = max(Index.items(), key=operator.itemgetter(1))[0]
                else:
                    comp_Index = {ind: Index[ind] for ind in comp_set}
                    k_t = max(comp_Index.items(), key=operator.itemgetter(1))[0]

                pulls[k_t] = pulls[k_t] + 1

                reward = self.generate_sample(k_t)

                #Update \mu_{k_t}
                sumReward[k_t] = sumReward[k_t] + reward
                empReward[k_t] = sumReward[k_t]/float(pulls[k_t])

                #Pseudo-reward updates
                # pseudoRewards = tables[k_t][reward-1, :] #(zero-indexed)
                pseudoRewards = np.array(tables[tables.pull==k_t][tables.reward==reward.values[0]-1]).reshape(-1)[2:]

                sumPseudoReward[:, k_t] = sumPseudoReward[:, k_t] + pseudoRewards
                empPseudoReward[:, k_t] = np.divide(sumPseudoReward[:, k_t], float(pulls[k_t]))

                #Diagonal elements of pseudorewards
                empPseudoReward[np.arange(numArms), np.arange(numArms)] = empReward

                #Update UCB+LCB indices:
                for k in range(numArms):
                    if(pulls[k] > 0):
                        #UCB index
                        Index[k] = empReward[k] + B[k]*np.sqrt(2. * np.log(t+1)/pulls[k])

                #Regret calculation
                if t == 0:
                    cucb_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    cucb_regret[t] = cucb_regret[t-1] + true_means_test[optArm] - true_means_test[k_t]

            avg_cucb_regret[iteration, :] = cucb_regret

        print(f'Num Pulls : CUCB: {pulls}')
        return avg_cucb_regret

    def C_TS(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_tsc_regret = np.zeros((num_iterations, T))

        beta = 4. #since sigma was taken as 2
        for iteration in tqdm(range(num_iterations)):
            TSC_pulls = np.zeros(numArms)

            TSC_empReward = np.zeros(numArms)
            TSC_sumReward = np.zeros(numArms)

            TSC_empReward[:] = np.inf

            TSC_empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            TSC_sumPseudoReward = np.zeros((numArms, numArms))

            TSC_empPseudoReward[:,:] = np.inf


            tsc_regret = np.zeros(T)

            for t in range(T):

                #add to set \ell for arms with pulls >t/K
                bool_ell = TSC_pulls >= (float(t - 1)/numArms)

                max_mu_hat = np.max(TSC_empReward[bool_ell])

                if TSC_empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = TSC_empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(TSC_empReward == max_mu_hat)[0][0]

                #Set of competitive arms - update through the run
                min_phi = np.min(TSC_empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                #Adding the argmax arm
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                if t < numArms:
                    k_t = t #%numArms
                else:
                    # Thompson Sampling
                    thompson = self.ThompsonSample(TSC_empReward, TSC_pulls, beta)
                    comp_values = {ind: thompson[ind] for ind in comp_set}
                    k_t = max(comp_values.items(), key=operator.itemgetter(1))[0]

                TSC_pulls[k_t] = TSC_pulls[k_t] + 1

                reward = self.generate_sample(k_t)

                # Update \mu_{k_t}
                TSC_sumReward[k_t] = TSC_sumReward[k_t] + reward
                TSC_empReward[k_t] = TSC_sumReward[k_t]/float(TSC_pulls[k_t])

                # Pseudo-reward updates
                # TSC_pseudoRewards = tables[k_t][reward-1, :] #(zero-indexed)
                TSC_pseudoRewards = np.array(tables[tables.pull==k_t][tables.reward==reward.values[0]-1]).reshape(-1)[2:]

                TSC_sumPseudoReward[:, k_t] = TSC_sumPseudoReward[:, k_t] + TSC_pseudoRewards
                TSC_empPseudoReward[:, k_t] = np.divide(TSC_sumPseudoReward[:, k_t], float(TSC_pulls[k_t]))

                # Regret calculation
                if t == 0:
                    tsc_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    tsc_regret[t] = tsc_regret[t-1] + true_means_test[optArm] - true_means_test[k_t]

            avg_tsc_regret[iteration, :] = tsc_regret

        print(f'Num Pulls : CTS  : {TSC_pulls}')
        return avg_tsc_regret

    def run(self, num_iterations=20, T=5000, algo='all'):

        if algo=='eg' or algo=='all' : avg_eg_regret = self.epsilon_greedy(num_iterations, T)
        if algo=='ucb' or algo=='all' : avg_ucb_regret = self.UCB(num_iterations, T)
        if algo=='ts' or algo=='all' : avg_ts_regret = self.TS(num_iterations, T)
        if algo=='gb' or algo=='all' : avg_gb_regret = self.GB(num_iterations, T)

        if algo=='ceg' or algo=='all' : avg_ceg_regret = self.C_epsilon_greedy(num_iterations, T)
        if algo=='cucb' or algo=='all' : avg_cucb_regret = self.C_UCB(num_iterations, T)
        if algo=='cts' or algo=='all' : avg_cts_regret = self.C_TS(num_iterations, T)
        if algo=='cgb' or algo=='all' : avg_cgb_regret = self.C_GB(num_iterations, T)


        # mean cumulative regret
        if algo=='eg' or algo=='all' : self.plot_av_eg = np.mean(avg_eg_regret, axis=0)
        if algo=='ucb' or algo=='all' : self.plot_av_ucb = np.mean(avg_ucb_regret, axis=0)
        if algo=='ts' or algo=='all' : self.plot_av_ts = np.mean(avg_ts_regret, axis=0)
        if algo=='gb' or algo=='all' : self.plot_av_gb = np.mean(avg_gb_regret, axis=0)
        if algo=='ceg' or algo=='all' : self.plot_av_ceg = np.mean(avg_ceg_regret, axis=0)
        if algo=='cucb' or algo=='all' : self.plot_av_cucb = np.mean(avg_cucb_regret, axis=0)
        if algo=='cts' or algo=='all' : self.plot_av_cts = np.mean(avg_cts_regret, axis=0)
        if algo=='cgb' or algo=='all' : self.plot_av_cgb = np.mean(avg_cgb_regret, axis=0)

        # std dev over runs
        if algo=='eg' or algo=='all' : self.plot_std_eg = np.sqrt(np.var(avg_eg_regret, axis=0))
        if algo=='ucb' or algo=='all' : self.plot_std_ucb = np.sqrt(np.var(avg_ucb_regret, axis=0))
        if algo=='ts' or algo=='all' : self.plot_std_ts = np.sqrt(np.var(avg_ts_regret, axis=0))
        if algo=='gb' or algo=='all' : self.plot_std_gb = np.sqrt(np.var(avg_gb_regret, axis=0))
        if algo=='ceg' or algo=='all' : self.plot_std_ceg = np.sqrt(np.var(avg_ceg_regret, axis=0))
        if algo=='cucb' or algo=='all' : self.plot_std_cucb = np.sqrt(np.var(avg_cucb_regret, axis=0))
        if algo=='cts' or algo=='all' : self.plot_std_cts = np.sqrt(np.var(avg_cts_regret, axis=0))
        if algo=='cgb' or algo=='all' : self.plot_std_cgb = np.sqrt(np.var(avg_cgb_regret, axis=0))

        self.save_data(algo)

    def edit_data(self):

        if self.exp_name == 'genre':
            # code only masks values as done in the paper
            genre_tables = pd.read_csv(f'preproc/{self.exp_name}s/{self.exp_name}_table.csv')
            p = self.p
            for genre in range(18):
                for row in range(5):
                    row_len = 18
                    genre_tables.loc[(genre_tables.pull==genre) & (genre_tables.reward==row), list(map(str, np.random.choice(np.arange(row_len), size=int(p*row_len), replace=False)))] = 5.
            # restore reference columns
            for genre in range(18):
                genre_tables.loc[genre_tables['pull']==genre, str(genre)] = np.arange(1,6)

            self.tables = genre_tables

        elif self.exp_name == 'movie':
            # code only pads entries as done in the paper
            movie_tables = pd.read_csv(f'preproc/{self.exp_name}s/{self.exp_name}_table.csv')
            pad_val = self.padval
            for movie in range(50): # top 50 movies picked in preproc
                movie_tables.loc[movie_tables.pull==movie, list(map(str, np.arange(50)))] += pad_val
            for i in range(50) :
                movie_tables[str(i)] = movie_tables[str(i)].clip(upper=5.0)
            for movie in range(50) :
                movie_tables.loc[movie_tables['pull']==movie, str(movie)] = np.arange(1,6)

            self.tables = movie_tables

        elif self.exp_name == 'book':
            book_tables = pd.read_pickle(f'preproc/{self.exp_name}s/book_tables_train.pkl')
            p = self.p
            pad_val = self.padval
            for book in range(25): # top 25 books picked in preproc
                for row in range(book_tables[book].shape[0]):
                    row_len = int(book_tables[book].shape[1])
                    book_tables[book][row][np.random.choice(np.arange(row_len), size= int(p*row_len),
                                                            replace=False)] = 5.

                book_tables[book] += pad_val
                book_tables[book][book_tables[book] > 5] = 5.

                book_tables[book][:, book] = np.arange(1,6)

            self.tables = book_tables

    def save_data(self, algo='all'):
        algorithms = ['eg','ucb', 'ts', 'gb', 'ceg', 'cucb', 'cts', 'cgb'] if algo=='all' else [algo]
        pathlib.Path(f'plot_arrays/{self.exp_name}s/').mkdir(parents=True, exist_ok=True)
        for alg in algorithms:
            np.save(f'plot_arrays/{self.exp_name}s/plot_av_{alg}_p{self.p:.2f}_pad{self.padval:.2f}',
                    getattr(self, f'plot_av_{alg}'))
            np.save(f'plot_arrays/{self.exp_name}s/plot_std_{alg}_p{self.p:.2f}_pad{self.padval:.2f}',
                    getattr(self, f'plot_std_{alg}'))

    def plot(self, num_iterations, T, algo='all'):
        spacing = 400
        # Means
        if algo=='eg' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_eg[::spacing], label='EpsilonGreedy', color='green', marker='*')
        if algo=='ucb' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_ucb[::spacing], label='UCB', color='red', marker='+')
        if algo=='ts' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_ts[::spacing], label='TS', color='yellow', marker='o')
        if algo=='gb' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_gb[::spacing], label='GB', color='magenta', marker='1')
        if algo=='ceg' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_ceg[::spacing], label='C-EpsilonGreedy', color='orange', marker='p')
        if algo=='cucb' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_cucb[::spacing], label='C-UCB', color='blue', marker='^')
        if algo=='cts' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_cts[::spacing], label='C-TS', color='black', marker='x')
        if algo=='cgb' or algo=='all' : plt.plot(range(0, T)[::spacing], self.plot_av_cgb[::spacing], label='C-GB', color='cyan', marker='2')

        # Confidence bounds
        if algo=='eg' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_eg + self.plot_std_eg)[::spacing],
                         (self.plot_av_eg - self.plot_std_eg)[::spacing], alpha=0.2, facecolor='g')
        if algo=='ucb' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_ucb + self.plot_std_ucb)[::spacing],
                         (self.plot_av_ucb - self.plot_std_ucb)[::spacing], alpha=0.2, facecolor='r')
        if algo=='ts' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_ts + self.plot_std_ts)[::spacing],
                         (self.plot_av_ts - self.plot_std_ts)[::spacing], alpha=0.2, facecolor='y')
        if algo=='gb' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_gb + self.plot_std_gb)[::spacing],
                         (self.plot_av_gb - self.plot_std_gb)[::spacing], alpha=0.2, facecolor='magenta')
        if algo=='ceg' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_ceg + self.plot_std_ceg)[::spacing],
                         (self.plot_av_ceg - self.plot_std_ceg)[::spacing], alpha=0.2, facecolor='orange')
        if algo=='cucb' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_cucb + self.plot_std_cucb)[::spacing],
                         (self.plot_av_cucb - self.plot_std_cucb)[::spacing], alpha=0.2, facecolor='b')
        if algo=='cts' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_cts + self.plot_std_cts)[::spacing],
                         (self.plot_av_cts - self.plot_std_cts)[::spacing], alpha=0.2, facecolor='k')
        if algo=='cgb' or algo=='all' : plt.fill_between(range(0, T)[::spacing], (self.plot_av_cgb + self.plot_std_cgb)[::spacing],
                         (self.plot_av_cgb - self.plot_std_cgb)[::spacing], alpha=0.2, facecolor='cyan')
        
        # Plot
        plt.legend()
        plt.grid(True, axis='y')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Cumulative Regret')
        # Save
        pathlib.Path('data/plots/').mkdir(parents=False, exist_ok=True)
        plt.savefig(f'data/plots/{self.exp_name}_p{self.p:.2f}_pad{self.padval:.2f}_iter{num_iterations}.pdf')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', dest='exp', type=str, default='genre', help="Experiment to run (genre, movie, book)")
    parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=20,
                        help="Number of iterations of each run")
    parser.add_argument('--T', dest='T', type=int, default=5000, help="Number of rounds")
    parser.add_argument('--p', dest='p', type=float, default=0.0, help="Fraction of table entries to mask")
    parser.add_argument('--padval', dest='padval', type=float, default=0.0, help="Padding value for table entries")
    parser.add_argument('--algo', dest='algo', type=str, default='all', help="Which algo to run")
    parser.add_argument('--true_means', dest='true_means', type=str, default='true_means_test', help="Which dataset to use")
    parser.add_argument('--random', dest='rand', type=bool, default=False, help='Whether to sample randomly, based on true means')
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    bandit_obj = algs(args.exp, p=args.p, padval=args.padval, true_means=args.true_means, rand=args.rand)
    bandit_obj.edit_data()
    bandit_obj.run(args.num_iterations, args.T, args.algo)
    bandit_obj.plot(args.num_iterations, args.T, args.algo)


if __name__ == '__main__':
    main(sys.argv)
