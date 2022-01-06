# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:39:12 2021

@author: acanlab
"""
import matplotlib.pyplot as plt
import random
import numpy as np
from Class_define_ESNSGA import myProblem
from Class_define_ESNSGA import myEvolution
#%%
path = 'C:\\Users\\acanlab\\Desktop\\sihan\\nsga\\MO-G-FJSP_P1.txt'
f = open(path, 'r')
line_cnt=0
index_of_map=0
total_operation=0

color = ['red','green','blue','orange','yellow','purple','gray','pink','brown','black']
for line in f:
    line_split = line.split()
    if len(line_split)==0:
        continue
    if line_cnt==0:
        N_jobs = int(line_split[0])
        N_machines = int(line_split[1])
        job_machine_operation_map = [[]for _ in range(N_jobs)]
        operation_arr = []
        job_cnt=[]*N_jobs
        machine_distance = [[0]*N_machines]*N_machines
        obj_matrix=[]
    elif line_cnt==1: #load job cnt
        job_cnt = [int(d) for d in line_split]
    elif line_cnt>=2 and line_cnt<7: # load objective (total n_machine lines)
        tmp = []*5
        tmp = [int(d) for d in line_split]
        obj_matrix.append(tmp)
    elif line_cnt>=7 and line_cnt<7+N_machines: #load machine distance
        #print(cnt)
        machine_distance[line_cnt-7] = [int(d) for d in line_split]
    else:
        index_of_line_split = 0
        #Get numbers of operation of each job
        N_operations = int(line_split[index_of_line_split]) 
        index_of_line_split +=1
        total_operation+=N_operations
        for i in range(N_operations):
            operation_arr.append([i,line_cnt-7]) #add to operation list for computing objective
            N_nums = int(line_split[index_of_line_split])
            tmp = [10000 for _ in range(N_machines)]
            for j in range(N_nums):   
                machine_index = int(line_split[index_of_line_split+1])-1
                operate_time = int(line_split[index_of_line_split+2])
                tmp[machine_index] = operate_time
                index_of_line_split += 2
            job_machine_operation_map[index_of_map].append(tmp)
            index_of_line_split += 1
        index_of_map += 1
    line_cnt+=1
f.close()
#%%

def split_Gene(Gene):
    job_batch = []
    tmp_batch_size = []
    cnt = 0
    #print(Gene)
    while cnt < len(Gene):
        if cnt==0:
            for i in range(len(job_cnt)): # job_cnt global variable
                job_batch.append(Gene[cnt])
                cnt+=1
        tmp_batch_size.append(Gene[cnt])
        cnt+=1
    batch_size = tmp_batch_size[:-1]
    schedule = tmp_batch_size[-1]
    cnt1=0
    batch_size_per_job=[]*len(job_batch)
    operation_processing=[]*len(job_batch)
    last_machine_operate=[]*len(job_batch)
    for job_cnt_tmp in job_batch:
        batch_set = []
        tmp_set1 = []
        tmp_set2 = []
        for i in range(cnt1,job_cnt_tmp+cnt1):
            batch_set.append(batch_size[i])
            tmp_set1.append(-1)
            tmp_set2.append(-1)
        batch_size_per_job.append(batch_set)
        operation_processing.append(tmp_set1)
        last_machine_operate.append(tmp_set2)
        cnt1+=job_cnt_tmp
    
    return job_batch, batch_size, schedule, batch_size_per_job, operation_processing, last_machine_operate
def Calculate(Gene):
    job_batch, batch_size, schedule, batch_size_per_job,operation_processing ,last_machine_operate = split_Gene(Gene)
    machine_nums = N_machines # global variable machine count
    job_nums = N_jobs #global variable job count
    operation_end_time = [0]*machine_nums # store the end time of last operation in each job
    machine_end_time = [0]*machine_nums # store last operation end time of each machine
    transportation_time = 0
    transfer_time = 0
    energy = 0
    for operation in schedule:
       
        job_index = operation[0]
        batch_index = operation[1]-1
        
        if last_machine_operate[job_index][batch_index] == -1: #each job for the first operation
            minima = 1000000
            mini_index = -1
            for i in range(machine_nums):
                time = job_machine_operation_map[job_index][0][i]*batch_size_per_job[job_index][batch_index]
                total_time = time + machine_end_time[i]
                if total_time < minima:
                    minima = total_time
                    mini_index = i
            operation_end_time[mini_index] = minima
            machine_end_time[mini_index] = minima
            last_machine_operate[job_index][batch_index] = mini_index
            
            
            operation_processing[job_index][batch_index] = 1
            transfer_time += obj_matrix[mini_index][0]
            energy += (obj_matrix[mini_index][3]*minima + obj_matrix[mini_index][4])
        else:
            minima = 1000000
            mini_index = -1
            
            op = operation_processing[job_index][batch_index]
            last_machine = last_machine_operate[job_index][batch_index]
            operate_time = operation_end_time[last_machine]
            for i in range(machine_nums):
                time = job_machine_operation_map[job_index][op][i]*batch_size_per_job[job_index][batch_index]
                if time!= 10000:
                    machine_time = time + machine_end_time[i]
                    op_time = time + operate_time
                    total_time = min(machine_time,op_time)
                    if total_time < minima:
                        minima = total_time
                        mini_index = i
            operation_end_time[mini_index] = minima
            machine_end_time[mini_index] = minima
            last_machine_operate[job_index][batch_index] = mini_index
            
            operation_processing[job_index][batch_index] += 1
            transportation_time += machine_distance[last_machine][mini_index]
            if last_machine!=mini_index:
                transfer_time += (obj_matrix[last_machine][1]+obj_matrix[mini_index][0])
            energy += obj_matrix[mini_index][3]*minima
            
            
    makespan = max(machine_end_time)
    return makespan, transfer_time, transportation_time, energy
def makespan(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return makespan
def transfer_time(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return transfer_time
def transportation_time(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return transportation_time
def energy(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return energy
def plot_gantt(feature):
    MGen=feature[0]
    OGen=feature[1]
    job_nums = N_jobs
    operation_nums = len(MGen)
    machine_nums = N_machines #Global variable for number of machines
    operation_end_time = [0]*machine_nums # store the end time of last operation in each job
    machine_end_time = [0]*machine_nums # store last operation end time of each machine
    last_machine_operate = [-1]*job_nums
    transfer_time = 0
    for i in range(len(MGen)):
        arg_index = np.argsort(OGen)
        operation_index = operation_arr[arg_index[i]][0]
        job_index = operation_arr[arg_index[i]][1]
        machine_index = MGen[arg_index[i]]
        span = job_machine_operation_map[job_index][operation_index][machine_index]

        c = color[job_index]
        if operation_index == 0:
            plt.barh(machine_index,span,left=machine_end_time[machine_index],color=c)
            plt.text(machine_end_time[machine_index]+span/4,machine_index,'J'+str(job_index)+'o'+str(operation_index),color='white')
            machine_end_time[machine_index] += span
            operation_end_time[machine_index] = machine_end_time[machine_index]
            last_machine_operate[job_index] = machine_index
        else:
            last_machine_index = last_machine_operate[job_index]
            # print(last_machine_index)
            operation_end = operation_end_time[last_machine_index]
            machine_end = machine_end_time[machine_index]
            transfer_time += machine_distance[last_machine_index][machine_index]
            if operation_end > machine_end:
                plt.barh(machine_index,span,left=operation_end,color=c)
                plt.text(operation_end+span/4,machine_index,'J'+str(job_index)+'o'+str(operation_index),color='white')
                machine_end_time[machine_index] = operation_end + span
                operation_end_time[machine_index] = machine_end_time[machine_index]
                last_machine_operate[job_index] = machine_index
            else:
                plt.barh(machine_index,span,left=machine_end_time[machine_index],color=c)
                plt.text(machine_end_time[machine_index]+span/4,machine_index,'J'+str(job_index)+'o'+str(operation_index),color='white')
                machine_end_time[machine_index] += span
                operation_end_time[machine_index] = machine_end_time[machine_index]
                last_machine_operate[job_index] = machine_index
    plt.show()
#%%
if __name__ == '__main__' :
    variables_range=[[(0,N_machines-1)]*total_operation,total_operation]
    operation_num_per_jobs = []
    for i in range(len(job_machine_operation_map)):
        operation_num_per_jobs.append(len(job_machine_operation_map[i])) 
    variable = []
    variable.append(job_cnt)
    variable.append(operation_num_per_jobs)
    fittness = [makespan, transfer_time, transportation_time, energy]
    problem = myProblem(num_of_variables=1, 
                      objectives=fittness, 
                      variables_range=variable,
                      operation_num_per_jobs=operation_num_per_jobs,
                      job_machine_operation_map = job_machine_operation_map)
    individual_1 = problem.generate_individual()
    individual_2 = problem.generate_individual()
    problem.calculate_objectives(individual_1)
    problem.calculate_objectives(individual_2)
    
    print("Evolutioin......")
    evo = myEvolution(problem, num_of_generations=200, num_of_individuals=1000)
    evol = evo.evolve()
    func=[]
    feature=[]
    for i in range(len(evol)):
        func.append(evol[i].objectives)
        feature.append(evol[i].features)
    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    function3 = [i[2] for i in func]
    function4 = [i[3] for i in func]
    plt.xlabel('maxspan', fontsize=15)
    plt.ylabel('transfer_time', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()

    print("End......")
#%% testing
    a, b ,c, d = Calculate(individual_1.features[0])
    test = [f(*individual_1.features) for f in fittness]
