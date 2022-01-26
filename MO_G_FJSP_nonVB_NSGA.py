# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:39:12 2021

@author: acanlab
"""
import matplotlib.pyplot as plt
import random
import numpy as np
from Class_define_NSGA_nonVB import myProblem
from Class_define_NSGA_nonVB import myEvolution
#%%
path = 'C:\\Users\\acanlab\\Desktop\\sihan\\nsga\\nsga2_data\\MO-G-FJSP_P1.fjs'
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
    elif line_cnt>=2 and line_cnt<2+N_machines: # load objective (total n_machine lines)
        tmp = []*5
        tmp = [int(d) for d in line_split]
        obj_matrix.append(tmp)
    elif line_cnt>=2+N_machines and line_cnt<2+N_machines+N_machines: #load machine distance
        #print(cnt)
        machine_distance[line_cnt-2-N_machines] = [int(d) for d in line_split]
    else:
        index_of_line_split = 0
        #Get numbers of operation of each job
        N_operations = int(line_split[index_of_line_split]) 
        index_of_line_split +=1
        total_operation+=N_operations
        for i in range(N_operations):
            operation_arr.append([i,line_cnt-2-N_machines-N_machines]) #add to operation list for computing objective
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
    batch_size = tmp_batch_size[:-2]
    schedule = tmp_batch_size[-2]
    machine_schedule = tmp_batch_size[-1]
    cnt1=0
    batch_size_per_job=[]*len(job_batch)
    operation_processing=[]*len(job_batch)
    last_machine_operate=[]*len(job_batch)
    last_operate_end_time=[]*len(job_batch)
    for job_cnt_tmp in job_batch:
        batch_set = []
        tmp_set1 = []
        tmp_set2 = []
        tmp_set3 = []
        for i in range(cnt1,job_cnt_tmp+cnt1):
            batch_set.append(batch_size[i])
            tmp_set1.append(-1)
            tmp_set2.append(-1)
            tmp_set3.append(0)
        batch_size_per_job.append(batch_set)
        operation_processing.append(tmp_set1)
        last_machine_operate.append(tmp_set2)
        last_operate_end_time.append(tmp_set3)
        cnt1+=job_cnt_tmp
    
    return job_batch, batch_size, schedule, batch_size_per_job, operation_processing, last_machine_operate, last_operate_end_time, machine_schedule
def Calculate(Gene):
    job_batch, batch_size, schedule, batch_size_per_job,operation_processing ,last_machine_operate, last_operate_end_time, machine_schedule = split_Gene(Gene)
    machine_nums = N_machines # global variable machine count
    # job_nums = N_jobs #global variable job count
    machine_end_time = [0]*machine_nums # store last operation end time of each machine
    transportation_time = 0
    transfer_time = 0
    energy = 0
    for operation in schedule:
       
        job_index = operation[0]
        batch_index = operation[1]-1
        if last_machine_operate[job_index][batch_index] == -1: #each job for the first operation
            machine_schedule_index = machine_schedule[batch_index][job_index][0]    
            time = job_machine_operation_map[job_index][0][machine_schedule_index]*batch_size_per_job[job_index][batch_index]
            machine_end_time[machine_schedule_index] += time
            last_operate_end_time[job_index][batch_index] = machine_end_time[machine_schedule_index]
            last_machine_operate[job_index][batch_index] = machine_schedule_index
            # plt.barh(mini_index,time,left=machine_end_time[mini_index],color=c)
            # plt.text(machine_end_time[mini_index]+time/4,mini_index,'J'+str(job_index)+'o'+str(0),color='white')
            
            operation_processing[job_index][batch_index] = 1
            transfer_time += obj_matrix[machine_schedule_index][0]
            energy += (obj_matrix[machine_schedule_index][3]*time + obj_matrix[machine_schedule_index][4])
        else:
            op = operation_processing[job_index][batch_index]
            machine_schedule_index = machine_schedule[batch_index][job_index][op]
            last_machine = last_machine_operate[job_index][batch_index]
            operate_time = last_operate_end_time[job_index][batch_index]
            time = job_machine_operation_map[job_index][op][machine_schedule_index]*batch_size_per_job[job_index][batch_index]
            machine_time = time + machine_end_time[machine_schedule_index]
            op_time = time + operate_time
            total_time = max(machine_time, op_time)
            last_operate_end_time[job_index][batch_index] = total_time
            machine_end_time[machine_schedule_index] = total_time
            last_machine_operate[job_index][batch_index] = machine_schedule_index
            # plt.barh(mini_index,time,left=minima-time,color=c)
            # plt.text(minima-time+time/4,mini_index,'J'+str(job_index)+'o'+str(operation_processing[job_index][batch_index]),color='white')
            operation_processing[job_index][batch_index] += 1
            transportation_time += machine_distance[last_machine][machine_schedule_index]
            if last_machine!=machine_schedule_index:
                transfer_time += (obj_matrix[last_machine][1]+obj_matrix[machine_schedule_index][0])
            energy += obj_matrix[machine_schedule_index][3]*total_time
            
            
    makespan = max(machine_end_time)
    return makespan, transfer_time, transportation_time, energy
def Makespan(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return makespan
def Transfer_time(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return transfer_time
def Transportation_time(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return transportation_time
def Energy(Gene):
    makespan, transfer_time, transportation_time, energy = Calculate(Gene)
    return energy
def plot_gantt(feature):
    job_batch, batch_size, schedule, batch_size_per_job,operation_processing ,last_machine_operate, last_operate_end_time, machine_schedule = split_Gene(feature)
    machine_nums = N_machines # global variable machine count
    # job_nums = N_jobs #global variable job count
    machine_end_time = [0]*machine_nums # store last operation end time of each machine
    transportation_time = 0
    transfer_time = 0
    energy = 0
    for operation in schedule:
       
        job_index = operation[0]
        batch_index = operation[1]-1
        c = color[job_index]
        if last_machine_operate[job_index][batch_index] == -1: #each job for the first operation
            machine_schedule_index = machine_schedule[batch_index][job_index][0]    
            time = job_machine_operation_map[job_index][0][machine_schedule_index]*batch_size_per_job[job_index][batch_index] 
            machine_end_time[machine_schedule_index] += time
            last_operate_end_time[job_index][batch_index] = machine_end_time[machine_schedule_index]
            last_machine_operate[job_index][batch_index] = machine_schedule_index
            plt.barh(machine_schedule_index,time,left=machine_end_time[machine_schedule_index]-time,color=c)
            plt.text(time/4,machine_schedule_index,'J'+str(job_index)+'o'+str(0),color='white')
            
            operation_processing[job_index][batch_index] = 1
            transfer_time += obj_matrix[machine_schedule_index][0]
            energy += (obj_matrix[machine_schedule_index][3]*time + obj_matrix[machine_schedule_index][4])
        else:
            op = operation_processing[job_index][batch_index]
            machine_schedule_index = machine_schedule[batch_index][job_index][op]
            last_machine = last_machine_operate[job_index][batch_index]
            operate_time = last_operate_end_time[job_index][batch_index]
            time = job_machine_operation_map[job_index][op][machine_schedule_index]*batch_size_per_job[job_index][batch_index]
            machine_time = time + machine_end_time[machine_schedule_index]
            op_time = time + operate_time
            total_time = max(machine_time, op_time)
            last_operate_end_time[job_index][batch_index] = total_time
            machine_end_time[machine_schedule_index] = total_time
            last_machine_operate[job_index][batch_index] = machine_schedule_index
            plt.barh(machine_schedule_index,time,left=total_time-time,color=c)
            plt.text(total_time-time+time/4,machine_schedule_index,'J'+str(job_index)+'o'+str(operation_processing[job_index][batch_index]),color='white')
            operation_processing[job_index][batch_index] += 1
            transportation_time += machine_distance[last_machine][machine_schedule_index]
            if last_machine!=machine_schedule_index:
                transfer_time += (obj_matrix[last_machine][1]+obj_matrix[machine_schedule_index][0])
            energy += obj_matrix[machine_schedule_index][3]*total_time
    plt.show()
#%%
if __name__ == '__main__' :
    variables_range=[[(0,N_machines-1)]*total_operation,total_operation]
    operation_num_per_jobs = []
    for i in range(len(job_machine_operation_map)):
        operation_num_per_jobs.append(len(job_machine_operation_map[i])) 
    variable = []
    job_cnt = [1]*N_jobs
    variable.append(job_cnt)
    variable.append(operation_num_per_jobs)
    fittness = [Makespan, Transfer_time]
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
    evo = myEvolution(problem, num_of_generations=30000, num_of_individuals=1000, mutation_schedule=[[0,120],[100,100],[200,80],[300,60],[400,40],[500,20],[600,10]])
    evol = evo.evolve()
    func=[]
    feature=[]
    for i in range(len(evol)):
        func.append(evol[i].objectives)
        feature.append(evol[i].features)
    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    # function3 = [i[2] for i in func]
    # function4 = [i[3] for i in func]
    plt.xlabel('maxspan', fontsize=15)
    plt.ylabel('transfer_time', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()

    print("End......")
#%% testing
    # a, b ,c, d, e, f, g, h = split_Gene(individual_1.features[0])
    # test = [f(*individual_1.features) for f in fittness]
    plot_gantt(feature[0][0])
