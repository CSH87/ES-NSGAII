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
def Calculate(MGen,OGen):
    operation_nums = len(MGen)
    machine_nums = N_machines #Global variable for number of machines
    maxspan = 0
    job_nums = N_jobs #Global variable for number of jobs
    operation_end_time = [0]*machine_nums # store the end time of last operation in each job
    machine_end_time = [0]*machine_nums # store last operation end time of each machine
    last_machine_operate = [-1]*job_nums
    transfer_time = 0
    for i in range(operation_nums):
        arg_index = np.argsort(OGen)
        operation_index = operation_arr[arg_index[i]][0]
        job_index = operation_arr[arg_index[i]][1]
        machine_index = MGen[arg_index[i]]
        # print(job_index)
        # print(operation_index)
        # print(machine_index)
        span = job_machine_operation_map[job_index][operation_index][machine_index]
        if operation_index == 0:
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
                machine_end_time[machine_index] = operation_end + span
                operation_end_time[machine_index] = machine_end_time[machine_index]
                last_machine_operate[job_index] = machine_index
            else:
                machine_end_time[machine_index] += span
                operation_end_time[machine_index] = machine_end_time[machine_index]
                last_machine_operate[job_index] = machine_index
    maxspan = max(machine_end_time)   
    return maxspan, transfer_time
def maxspan(MGen,OGen):
    maxspan, transfer_time = Calculate(MGen, OGen)
    return maxspan
def transfer_time(MGen,OGen):
    maxspan, transfer_time = Calculate(MGen, OGen)
    return transfer_time
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
    inputs = []
    inputs.append(job_cnt)
    inputs.append(operation_num_per_jobs)
    problem = myProblem(num_of_variables=1, 
                      objectives=[maxspan, transfer_time], 
                      variables_range=inputs,
                      operation_num_per_jobs=operation_num_per_jobs,
                      job_machine_operation_map = job_machine_operation_map)
    i = problem.generate_individual()
    j = problem.generate_individual()
    problem.calculate_objectives(i)
    problem.calculate_objectives(j)
    
    print("Evolutioin......")
    evo = myEvolution(problem, num_of_generations=1000, num_of_individuals=1000)
    evol = evo.evolve()
    func=[]
    feature=[]
    for i in range(len(evol)):
        func.append(evol[i].objectives)
        feature.append(evol[i].features)
    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    plt.xlabel('transfer time', fontsize=15)
    plt.ylabel('maxspan', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()

    print("End......")
#%% testing
"""
job=operation_num_per_jobs
operation_index=0
job_index=0
for times in range(variables_range[1]): #for all operation
    if job[job_index]==operation_index:
        operation_index=0
        job_index+=1
    machine_index = i.features[0][times]
    tmp = job_machine_operation_map[job_index][operation_index][machine_index]
    print('tmp ' + str(tmp))
    if tmp ==10000:
        print("test1")
        N_machine = variables_range[0][0][1]+1
        for m in range(N_machine): #find the machine can operate this operation
            print("m "+str(m) +" job_index "+str(job_index)+" operation_index "+str(operation_index) +" test1 "+str(job_machine_operation_map[job_index][operation_index][m]))    
            if job_machine_operation_map[job_index][operation_index][m] != 10000:
                    i.features[0][times] = m
                    print('m ' + str(m))
                    print('index '+ str(machine_index))
                    break
    operation_index +=1
problem.calculate_objectives(i)

time = []
cnt=0
for job_index in range(len(job_machine_operation_map)):
    tmp = operation_num_per_jobs[job_index]
    for i in range(tmp):
        machine_index = feature[9][0][cnt]
        cnt+=1
        time.append(job_machine_operation_map[job_index][i][machine_index])
"""
plot_gantt(feature[3])
arg_index = np.argsort(feature[3][1])
for i in arg_index:
    print(feature[3][1][i])