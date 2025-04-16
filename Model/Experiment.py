# import the model
from elfarclass import ElFarolBar

# import the main statistical and data-management libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import utilities libraries
import os
import time

class Experiment():
    def __init__(self, n_rip, n_exp, charts) -> None:

        # esperiment parameters
        self.n_rep = n_rip
        self.n_exp = n_exp
        self.charts = charts

        # fixed parameters
        self.num_weeks = 200
        self.respect_the_max=True
        self.num_agents = 2000
        self.num_contagious_agents = 50
        self.capacity=self.num_agents

        # sampled parameters
        #self.range_num_agents = range(20, 1001)
        self.range_threshold = np.arange(0.01, 1.01, 0.01)
        self.range_contagious_threshold = np.arange(0.01, 1.01, 0.01)
        self.range_contagious_duration = range(1, 11)
        self.range_people_memory_weight = np.arange(0.01, 1.01, 0.01)
        self.range_contagious_thresholdNotPresent = np.arange(0.01, 1.01, 0.01)
        self.range_SIR_AgentsRecoveryTime = range(1,11)

        # result storage dictionary
            
        self.data_dict = {
            'seed': [],
            'n_exp': [],
            'n_sim': [],
            'num_weeks': [],
            'num_agents': [],
            'capacity': [],
            'threshold': [],
            'contagious_threshold': [],
            'contagious_duration': [],
            'SIR_AgentsRecoveryTime': [],
            'people_memory_weight': [],
            'contagious_thresholdNotPresent': [],
            'num_contagious_agents': [],
            'mean_attendance': [],
            'mean_contagious': [],
            'mean_present_contagious': [],
            'std_attendance': [],
            'std_contagious': [],
            'std_present_contagious': [],
            'argmax_acft_attendance': [],
            'max_acft_attendance': [],
            'argmax_acft_contagious': [],
            'max_acft_contagious': []
        }


    def single_run(self, n_sim):

        def store_results(self, seed, n_sim, threshold, contagious_threshold, contagious_duration, people_memory_weight, contagious_thresholdNotPresent,
                          SIR_AgentsRecoveryTime, mean_attendance, mean_contagious, mean_present_contagious, std_attendance, std_contagious, std_present_contagious,
                          argmax_acft_attendance, max_acft_attendance, argmax_acft_contagious, max_acft_contagious):
            self.data_dict['seed'].append(seed)            
            self.data_dict['n_exp'].append(self.n_exp)
            self.data_dict['n_sim'].append(n_sim)
            self.data_dict['num_weeks'].append(self.num_weeks)
            self.data_dict['num_agents'].append(self.num_agents)
            self.data_dict['num_contagious_agents'].append(self.num_contagious_agents)
            self.data_dict['capacity'].append(self.capacity)
            self.data_dict['threshold'].append(threshold)
            self.data_dict['contagious_threshold'].append(contagious_threshold)
            self.data_dict['contagious_duration'].append(contagious_duration)
            self.data_dict['SIR_AgentsRecoveryTime'].append(SIR_AgentsRecoveryTime)
            self.data_dict['people_memory_weight'].append(people_memory_weight)
            self.data_dict['contagious_thresholdNotPresent'].append(contagious_thresholdNotPresent)
            self.data_dict['mean_attendance'].append(mean_attendance) 
            self.data_dict['mean_contagious'].append(mean_contagious)
            self.data_dict['mean_present_contagious'].append(mean_present_contagious)
            self.data_dict['std_attendance'].append(std_attendance) 
            self.data_dict['std_contagious'].append(std_contagious)
            self.data_dict['std_present_contagious'].append(std_present_contagious)
            self.data_dict['argmax_acft_attendance'].append(argmax_acft_attendance)
            self.data_dict['max_acft_attendance'].append(max_acft_attendance)
            self.data_dict['argmax_acft_contagious'].append(argmax_acft_contagious)
            self.data_dict['max_acft_contagious'].append(max_acft_contagious)


        # sampling the parameters' value used to explore the behaviour of the model
        #num_agents = num_agents#np.random.choice(range_num_agents)
        threshold = round(np.random.choice(self.range_threshold),2)
        contagious_threshold = round(np.random.choice(self.range_contagious_threshold),2)
        contagious_duration = np.random.choice(self.range_contagious_duration)
        people_memory_weight = round(np.random.choice(self.range_people_memory_weight),2)
        contagious_thresholdNotPresent = round(np.random.choice(self.range_contagious_thresholdNotPresent),2)
        SIR_AgentsRecoveryTime = np.random.choice(self.range_SIR_AgentsRecoveryTime)
        seed = np.random.randint(0,99999999)

        # initialize the model
        bar = ElFarolBar(seed = seed,
                         num_agents=self.num_agents, 
                         num_contagious_agents=self.num_contagious_agents,
                         capacity=self.capacity,
                         threshold=threshold,
                         contagious_threshold=contagious_threshold,
                         contagious_duration=contagious_duration,
                         people_memory_weight=people_memory_weight,
                         contagious_thresholdNotPresent=contagious_thresholdNotPresent,
                         Use_SIR=True,
                         SIR_AgentsRecoveryTime=SIR_AgentsRecoveryTime,
                         debugCSV=False)

        results = bar.simulate(num_weeks=self.num_weeks, respect_the_max=self.respect_the_max)
        attendance_history,contagious_history, present_contagious_history = results[1], results[2], results[3]
        t_remove = int(0.5 * self.num_weeks) # to avoid to count the transition phase
        attendance_history, contagious_history, present_contagious_history = attendance_history[t_remove:], contagious_history[t_remove:], present_contagious_history[t_remove:]
        mean_attendance, std_attendance = np.mean(attendance_history), np.std(attendance_history)
        mean_contagious, std_contagious= np.mean(contagious_history), np.std(contagious_history)
        mean_present_contagious, std_present_contagious = np.mean(present_contagious_history), np.std(present_contagious_history)
        argmax_acft_attendance, max_acft_attendance = self.get_periodicity(attendance_history)
        argmax_acft_contagious, max_acft_contagious = self.get_periodicity(contagious_history)

        store_results(self, seed, n_sim, threshold, contagious_threshold, contagious_duration, people_memory_weight, contagious_thresholdNotPresent, 
                      SIR_AgentsRecoveryTime, mean_attendance, mean_contagious, mean_present_contagious, std_attendance, std_contagious, std_present_contagious,
                      argmax_acft_attendance, max_acft_attendance, argmax_acft_contagious, max_acft_contagious)

        name_pic = "exp_" + str(self.n_exp) + "_" + str(n_sim)
        if self.charts: bar.chartSave(experiment = name_pic) 

    def store_experiment(self):

        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        baseDir = "/OutputCSV/"
        name_file = f"output_exp_{self.n_exp}.csv"

        df = pd.DataFrame(self.data_dict)
        df.to_csv(folder_path + baseDir + name_file)


    def get_periodicity(self, ts):
        if all(ts[i] == ts[i-1] for i in range(1, len(ts))): # all the elements are equal, so there is no periodicity
            return 0,0
        else: # otherwise, study the periodicity
            acf = sm.tsa.acf(ts, nlags=50)
            acf = acf[2:]
            max_acf, argmax_acf = max(acf), np.argmax(acf)
            return argmax_acf + 2, max_acf # +2 because I deleted the first two elements


def main():
    start_time = time.time()

    exp = Experiment(n_rip=3000, n_exp=7, charts = False) # experiment run

    for n_sim in range(exp.n_rep):
        elapsed_time, unit = time.time() - start_time, 'seconds'
        if elapsed_time > 60:
            elapsed_time, unit = elapsed_time / 60, 'minutes'
            if elapsed_time > 60:
                elapsed_time, unit = elapsed_time / 60, 'hours'
        if n_sim % 10 == 0: print(f'simulation {n_sim} of {exp.n_rep} in {elapsed_time:.2f} {unit}', end='\r')
        if n_sim % 50: exp.store_experiment()
        exp.single_run(n_sim)
    exp.store_experiment()
    print(f'Experiment {exp.n_exp} finished!')

if __name__ == "__main__":
    main()