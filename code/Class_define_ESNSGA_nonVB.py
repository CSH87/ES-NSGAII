# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:25:35 2021

@author: acanlab
"""
from nsga2.utils import NSGA2Utils
from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import random
import numpy as np
from nsga2.individual import Individual
from nsga2.population import Population
import copy
import queue
#%%
class myIndividual(Individual):
    def __init__(self):
        super(myIndividual,self).__init__()
        self.features = []
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            for x in range(len(self.features)):
                for y in range(len(self.features[x])):
                    if self.features[x][y] != other.features[x][y]:
                        # print(self.features[x][y])
                        return False
            return True
        return False
    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)
#%%
class myProblem(Problem):
    def __init__(self, objectives, num_of_variables, variables_range, operation_num_per_jobs, job_machine_operation_map, objective_obj, expand=True, same_range=False):
        super(myProblem,self).__init__(objectives, num_of_variables, variables_range, expand, same_range)
        self.operation_num_per_jobs = operation_num_per_jobs
        self.job_machine_operation_map = job_machine_operation_map
        self.objective_obj = objective_obj
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range
    def generate_machine_schedule(self,job_nums):
        tmp = self.variables_range[2]
        schedule = []
        for job_index in range(job_nums):
            operate_nums = self.variables_range[1][job_index]
            tmp_schedule = []
            for operate in range(operate_nums):
                #set each operation operate on machine in random
                operate_machine_nums = len(tmp[job_index][operate])
                index = random.randint(0,operate_machine_nums-1)
                tmp_schedule.append(tmp[job_index][operate][index])
            schedule.append(tmp_schedule)
        return schedule
    def generate_individual(self):
        individual = myIndividual()
        job_nums = len(self.variables_range[0])
        tmp_feature = []
        for i in range(job_nums):
            threshold = random.random()
            if self.variables_range[0][i] <2:
                tmp_feature.append(1)
            else:
                if threshold >0.5:
                    tmp_feature.append(1)
                else:
                    tmp_feature.append(2)
        schedule = []
        machine_schedule = []
        for i in range(job_nums):
            if tmp_feature[i]==1:
                tmp_feature.append(self.variables_range[0][i])
                for j in range(self.variables_range[1][i]):
                    schedule.append([i,1])
            else:
                batch1 = random.randint(1,self.variables_range[0][i]-1)
                batch2 = self.variables_range[0][i] - batch1
                tmp_feature.append(batch1)
                tmp_feature.append(batch2)
                for j in range(self.variables_range[1][i]):
                    schedule.append([i,1])
                #for j in range(self.variables_range[1][i]):
                    schedule.append([i,2])
        machine_schedule.append(self.generate_machine_schedule(job_nums))
        random.shuffle(schedule)
        tmp_feature.append(schedule)
        tmp_feature.append(machine_schedule)
        individual.features.append(tmp_feature)
        return individual
    
    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]
#%%
class myUtils(NSGA2Utils):

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):
        super(myUtils, self).__init__(problem, num_of_individuals,
                     num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            #print(individual.objectives)
            population.append(individual)
        return population
    def create_children(self, population, num_mutate):
        children = []
        #print("len : " + str(len(population)))
        child_mutate_set = self.__mutate(population, num_mutate)
        child_reseeding_set = self.__reseeding(population)
        for child_mutate in child_mutate_set:
            self.problem.calculate_objectives(child_mutate)
        for child_reseeding in child_reseeding_set:
            self.problem.calculate_objectives(child_reseeding)
            
        children.extend(child_mutate_set)
        children.extend(child_reseeding_set)
        
        return children
    def __mutate(self, population, num_mutate):
        half_population = int(len(population)/2)
        front_num = 0
        children = Population()
        mutate_population = copy.deepcopy(population)
        # select half best individual
        while len(children) < half_population:
            if len(children)+len(mutate_population.fronts[front_num]) <= half_population:
                children.extend(mutate_population.fronts[front_num])
            else:
                children.extend(mutate_population.fronts[front_num][0:half_population-len(children)])
            front_num += 1
        # half_population = int(self.num_of_individuals/2)
        # while len(children) < half_population:
        #     children.append(mutate_population.fronts[0][0])
        # mutate half best individual
        for individual in children:
            cnt_mutate = 0
            #print(individual)
            operation_index = len(individual.features[0][-2])-1
            while cnt_mutate < num_mutate :
                index1 = random.randint(0,operation_index)
                index2 = random.randint(0,operation_index)
                  
                if individual.features[0][-2][index1]!=individual.features[0][-2][index2]:
                    tmp = copy.deepcopy(individual.features[0][-2][index1])
                    individual.features[0][-2][index1] = copy.deepcopy(individual.features[0][-2][index2])
                    individual.features[0][-2][index2] = tmp
                    cnt_mutate+=1
            makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2 = self.problem.objective_obj.Calculate(*individual.features)
            threshold = random.random()
            cnt_machine = self.problem.variables_range[2]
            if threshold < 0 :
                
                for batch in range(len(individual.features[0][-1])):
                    for job in range(len(individual.features[0][-1][batch])):
                        for operation in range(len(individual.features[0][-1][batch][job])):
                            if individual.features[0][-1][batch][job][operation] == makespan_index:
                                threshold = random.random()
                                if threshold < 0.5 :
                                    machine_nums = len(cnt_machine[job][operation])
                                    machine_index = random.randint(0,machine_nums-1)
                                    individual.features[0][-1][batch][job][operation] = cnt_machine[job][operation][machine_index]
            else:
                for batch in range(len(individual.features[0][-1])):
                    for job in range(len(individual.features[0][-1][batch])):
                        operate_size = len(individual.features[0][-1][batch][job])
                        mutate_flag = 0
                        # mutate_times = operate_size//2
                        r = random.random()
                        threshold = 0.5
                        if r < threshold:
                        # while mutate_flag<=mutate_times:
                            operate_index = random.randint(0,operate_size-1)
                            cnt_machine_size = len(cnt_machine[job][operate_index])
                            machine_index = random.randint(0,cnt_machine_size-1)
                            if machine_index != individual.features[0][-1][batch][job][operate_index]:
                                individual.features[0][-1][batch][job][operate_index] = cnt_machine[job][operate_index][machine_index]
                                mutate_flag += 1
        return children
    def __reseeding(self, population):
        half_population = int(self.num_of_individuals/2) # reseed half of initial population 
        children = Population()
        for _ in range(half_population):
            individual = self.problem.generate_individual()
            children.append(individual)
        return children
    def tabu_search(self,individual):
        tabu_list_len = 5
        tabu_list = []
        best_individual = individual
        neighbor_searched = individual
        best_neighbor = [individual]
        iteration = 5
        pop_index = 0 
        for i in range(iteration):
            #print("tabu searching ...")
            #print("iteration : "+str(i))
            neighbor_searched, pop_index = self.tabu_search_iteration(neighbor_searched,
                                                           best_individual,
                                                           best_neighbor,tabu_list,
                                                           tabu_list_len,
                                                           pop_index)
            best_neighbor = [neighbor_searched]
        individual = best_individual
        
        
    def tabu_search_iteration(self,individual,best_individual,best_neighbor,tabu_list,
                              tabu_list_len,pop_index):
        machine_schedule = individual.features[0][-1]
        cnt_machine = self.problem.variables_range[2]
        index = 0
        for batch in range(len(machine_schedule)):
            for job in range(len(machine_schedule[batch])):
                for operation in range(len(machine_schedule[batch][job])):
                    #create one neighbor (change one machine schedule of one operation)
                    machine_nums = len(cnt_machine[job][operation])
                    machine_index = random.randint(0,machine_nums-1)
                    tmp_individual = copy.deepcopy(individual)
                    tmp_individual.features[0][-1][batch][job][operation] = cnt_machine[job][operation][machine_index] 
                    
                    # create another kind of neighbor (swap operation schedule one times)
                    operation_index = len(individual.features[0][-2])-1
                    index1 = 0
                    index2 = 0
                    while index1 == index2:
                        index1 = random.randint(0,operation_index)
                        index2 = random.randint(0,operation_index)
                    tmp_individual2 = copy.deepcopy(individual)
                    tmp = copy.deepcopy(tmp_individual2.features[0][-2][index1])
                    tmp_individual2.features[0][-2][index1] = copy.deepcopy(tmp_individual2.features[0][-2][index2])
                    tmp_individual2.features[0][-2][index2] = tmp
                    #create third neighbor (change schedule like first neighbor with neightbor2)
                    tmp_individual3 = copy.deepcopy(tmp_individual2)
                    tmp_individual3.features[0][-1][batch][job][operation] = cnt_machine[job][operation][machine_index]
                    
                    self.problem.calculate_objectives(tmp_individual)
                    self.problem.calculate_objectives(tmp_individual2)
                    self.problem.calculate_objectives(tmp_individual3)
                    if tmp_individual.dominates(best_neighbor[index]):
                        #print("add neighbor")
                        index+=1
                        best_neighbor.append(tmp_individual)
                    if tmp_individual2.dominates(best_neighbor[index]):
                        #print("add neighbor")
                        index+=1
                        best_neighbor.append(tmp_individual2)
                    if tmp_individual3.dominates(best_neighbor[index]):
                        #print("add neighbor")
                        index+=1
                        best_neighbor.append(tmp_individual3)
        tabu_list_checked = False
        #print("checking")
        while not tabu_list_checked:
            if best_neighbor[index].dominates(best_individual):
                #print("update individual")
                best_individual = copy.deepcopy(best_neighbor[index])
                if best_individual not in tabu_list:
                    if len(tabu_list) == tabu_list_len:
                        tabu_list[pop_index] = best_individual
                        pop_index += 1
                        if pop_index == tabu_list_len:
                            pop_index = 0
                    else:
                        tabu_list.append(best_individual)
                tabu_list_checked=True
            else:
                if best_neighbor[index] not in tabu_list:
                    if len(tabu_list) == tabu_list_len:
                        tabu_list[pop_index] = best_neighbor[index]
                        pop_index += 1
                        if pop_index == tabu_list_len:
                            pop_index = 0
                    else:
                        tabu_list.append(best_neighbor[index])
                    tabu_list_checked=True
                    #any((1, 1) in item for item in q.queue)
                else:
                    index-=1
            if index == -1:
                break
        return copy.deepcopy(best_neighbor[index]),pop_index
    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)
        # print("poplen : "+str(len(population)))
    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant
        return best
    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2*u)**(1/(self.mutation_param + 1)) - 1
        return u, 1 - (2*(1-u))**(1/(self.mutation_param + 1))
    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10**9
                front[solutions_num-1].crowding_distance = 10**9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num-1):
                    front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m])/scale
#%%
class myEvolution(Evolution):

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5,mutation_schedule=[[0,20],[50,15],[100,10]]):
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mutation_schedule = mutation_schedule
        self.utils = myUtils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
    def evolve(self):
        self.population = self.utils.create_initial_population()
        #檢查到這
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        num_mutate = self.mutation_schedule[0][1]
        mutate_index = 1
        max_search_nums = 100
        returned_population = None
        min_makespan = np.inf
        for i in range(self.num_of_generations):
            print('generation : ' + str(i))
            children = self.utils.create_children(self.population,num_mutate)
            # self.population = Population()
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            # self.utils.fast_nondominated_sort(children)
            for individual in self.population:
                if individual.objectives[0] < min_makespan:
                    min_makespan = individual.objectives[0]
                    print('Makespan : ' +str(min_makespan))
            # for individual in children:
            #     if individual.objectives[0] < min_makespan:
            #         min_makespan = individual.objectives[0]
            #         print('Makespan : ' +str(min_makespan))
            cnt=0
            # tabu_search_rate = 100
            # if ((i+1) % tabu_search_rate)==0:
            #     for individual in self.population.fronts[0]:
            #         print("tabu searching")
            #         print("individual : " + str(cnt))
            #         self.utils.tabu_search(individual)
            #         cnt+=1
            #         if cnt==max_search_nums:
            #             break
            new_population = Population()
            front_num = 0
            # print(self.population.fronts)
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals :
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            returned_population = copy.deepcopy(self.population)

            self.population = copy.deepcopy(new_population)
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            if mutate_index < len(self.mutation_schedule) and i == self.mutation_schedule[mutate_index][0]:
                num_mutate = self.mutation_schedule[mutate_index][1]
                mutate_index+=1
        cnt=0
        # for individual in returned_population.fronts[0]:
        #     print("tabu searching")
        #     print("individual : " + str(cnt))
        #     self.utils.tabu_search(individual)
        #     cnt+=1
        #     if cnt==max_search_nums:
        #         break
        #     if individual.objectives[0] < min_makespan:
        #         min_makespan = individual.objectives[0]
        #         print('Makespan : ' +str(min_makespan))
        return returned_population.fronts[0], min_makespan
class objective_calculation:
    def __init__(self, job_cnt, machine_cnt, job_machine_operation_map, obj_matrix, machine_distance,color):
        self.job_cnt = job_cnt
        self.machine_cnt = machine_cnt
        self.job_machine_operation_map = job_machine_operation_map
        self.obj_matrix = obj_matrix
        self.machine_distance = machine_distance
        self.color = color
    def split_Gene(self,Gene):
        job_batch = []
        tmp_batch_size = []
        cnt = 0
        #print(Gene)
        while cnt < len(Gene):
            if cnt==0:
                for i in range(len(self.job_cnt)): 
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
                tmp_set1.append(0)
                tmp_set2.append(-1)
                tmp_set3.append(0)
            batch_size_per_job.append(batch_set)
            operation_processing.append(tmp_set1)
            last_machine_operate.append(tmp_set2)
            last_operate_end_time.append(tmp_set3)
            cnt1+=job_cnt_tmp
        
        return job_batch, batch_size, schedule, batch_size_per_job, operation_processing, last_machine_operate, last_operate_end_time, machine_schedule
    def Calculate(self,Gene):
        job_batch, batch_size, schedule, batch_size_per_job,operation_processing ,last_machine_operate, last_operate_end_time, machine_schedule = self.split_Gene(Gene)
        machine_nums = self.machine_cnt 
        # job_nums = N_jobs #global variable job count
        machine_end_time = [0]*machine_nums # store last operation end time of each machine
        machine_blank_space = [[0] for _ in range(machine_nums) ] # store each machine blank space
        schedule_results = [[] for _ in range(machine_nums)] #schedule results version 1
        schedule_results_v2 = copy.deepcopy(machine_schedule)#schedule results version 2
        transportation_time = 0
        transfer_time = 0
        energy = 0
        for operation in schedule:
           
            job_index = operation[0]
            batch_index = operation[1]-1
            op = operation_processing[job_index][batch_index]
            machine_schedule_index = machine_schedule[batch_index][job_index][op]
            time = self.job_machine_operation_map[job_index][op][machine_schedule_index]*batch_size_per_job[job_index][batch_index]
            last_machine = last_machine_operate[job_index][batch_index]
            operate_time = last_operate_end_time[job_index][batch_index]
            machine_blank_space_size = len(machine_blank_space[machine_schedule_index])
            find_blank = False
            # for blank_index in range(1,machine_blank_space_size):
            #     blank_start = machine_blank_space[machine_schedule_index][blank_index][0]
            #     blank_end = machine_blank_space[machine_schedule_index][blank_index][1]
            #     blank_size = blank_end-blank_start
            #     if blank_size >= time and blank_start>=operate_time:
            #         find_blank = True
            #         total_time = blank_start + time
            #         machine_blank_space[machine_schedule_index][blank_index][0] = total_time
            #         last_operate_end_time[job_index][batch_index] = total_time
            #         last_machine_operate[job_index][batch_index] =machine_schedule_index
            #         schedule_results[machine_schedule_index].append([job_index,
            #                                                           batch_index,
            #                                                           op,
            #                                                           blank_start,
            #                                                           time])
            #         schedule_results_v2[batch_index][job_index][op] = \
            #             [machine_schedule_index,                                                                         
            #               blank_start,time]
            #         break
            #     elif blank_size >= time and blank_start<operate_time:
            #         if blank_size >= time+operate_time:
            #             find_blank = True
            #             total_time = operate_time + time
            #             machine_blank_space[machine_schedule_index][blank_index][1] = operate_time
            #             last_operate_end_time[job_index][batch_index] = total_time
            #             last_machine_operate[job_index][batch_index] =machine_schedule_index
            #             schedule_results[machine_schedule_index].append([job_index,
            #                                                               batch_index,
            #                                                               op,
            #                                                               operate_time,
            #                                                               time])
            #             schedule_results_v2[batch_index][job_index][op] = \
            #                 [machine_schedule_index,                                                                         
            #                   operate_time,time]
            #             break
            if not find_blank:
                machine_time = time + machine_end_time[machine_schedule_index]
                op_time = time + operate_time
                total_time = max(machine_time, op_time)
                last_operate_end_time[job_index][batch_index] = total_time
                machine_end_time[machine_schedule_index] = total_time
                last_machine_operate[job_index][batch_index] = machine_schedule_index
                if machine_time == total_time:
                    machine_blank_space[machine_schedule_index][0] = total_time
                    schedule_results[machine_schedule_index].append([job_index,
                                                                     batch_index,
                                                                     op,
                                                                     machine_time-time,
                                                                     time])
                    schedule_results_v2[batch_index][job_index][op] = \
                        [machine_schedule_index,                                                                         
                         machine_time-time,time]
                else:
                    machine_blank_space[machine_schedule_index].append([machine_time-time,op_time-time]) 
                    schedule_results[machine_schedule_index].append([job_index,
                                                                     batch_index,
                                                                     op,
                                                                     op_time-time,
                                                                     time])
                    schedule_results_v2[batch_index][job_index][op] = \
                        [machine_schedule_index,                                                                         
                         op_time-time,time]
                        
            operation_processing[job_index][batch_index] += 1
            if last_machine!=-1:
                transportation_time += self.machine_distance[last_machine][machine_schedule_index]
                if last_machine!=machine_schedule_index:
                    transfer_time += (self.obj_matrix[last_machine][1]+self.obj_matrix[machine_schedule_index][0])
            energy += self.obj_matrix[machine_schedule_index][3]*total_time
            
            
            
     
                
                
        makespan = max(machine_end_time)
        makespan_index = np.argmax(machine_end_time)
        for machine in schedule_results:
            machine.sort(key = lambda x:x[3])
        return makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2
    def Makespan(self,Gene):
        makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2 = self.Calculate(Gene)
        return makespan
    def Transfer_time(self,Gene):
        makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2 = self.Calculate(Gene)
        return transfer_time
    def Transportation_time(self,Gene):
        makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2 = self.Calculate(Gene)
        return transportation_time
    def Energy(self,Gene):
        makespan, transfer_time, transportation_time, energy, makespan_index, schedule_results, schedule_results_v2 = self.Calculate(Gene)
        return energy
    def plot_gantt(self,schedule_results):
        for machine_index in range(len(schedule_results)):
            for schedule in schedule_results[machine_index]:
                job_index = schedule[0]
                batch_index = schedule[1]
                operation_index = schedule[2]
                start_time = schedule[3]
                time = schedule[4]
                c = self.color[job_index]
                plt.barh(machine_index,time,left=start_time,color=c)
                plt.text(time/4+start_time,machine_index,'J'+str(job_index)+'o'+str(operation_index),color='white')         
            # plt.barh(machine_schedule_index,time,left=machine_end_time[machine_schedule_index]-time,color=c)
            # plt.text(time/4,machine_schedule_index,'J'+str(job_index)+'o'+str(0),color='white')
                
                
        plt.show()
    def check_schedule(self,schedule_results,schedule_results_v2):
        self.check_schedule1(schedule_results)
        self.check_schedule2(schedule_results_v2)
        print("Schedule is Valid !!")
    def check_schedule1(self,schedule_results):
        for machine in schedule_results:
            for machine_schedule_index1 in range(len(machine)):
                for machine_schedule_index2 in range(machine_schedule_index1+1,len(machine)):
                    start_time1 = machine[machine_schedule_index1][3]
                    time1 = machine[machine_schedule_index1][4]
                    start_time2 = machine[machine_schedule_index2][3]
                    if start_time1 + time1 > start_time2:
                        print("index1: " + str(machine_schedule_index1))
                        print("index2: " + str(machine_schedule_index2))
                        print("False")
                        return 
    def check_schedule2(self,schedule_results_v2):
        for batch in schedule_results_v2:
            for job in batch:
                for operation_index1 in range(len(job)):
                    for operation_index2 in range(operation_index1+1,len(job)):
                        start_time1 = job[operation_index1][1]
                        time1 = job[operation_index1][2]
                        start_time2 = job[operation_index2][1]
                        if start_time1 + time1 > start_time2:
                            print("index1: " + str(operation_index1))
                            print("index2: " + str(operation_index2))
                            print("False")
                            return 
        
