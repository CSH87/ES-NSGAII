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
    def __init__(self, objectives, num_of_variables, variables_range, operation_num_per_jobs, job_machine_operation_map, expand=True, same_range=False):
        super(myProblem,self).__init__(objectives, num_of_variables, variables_range, expand, same_range)
        self.operation_num_per_jobs = operation_num_per_jobs
        self.job_machine_operation_map = job_machine_operation_map
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range
    def valid_individual(self,individual):
        j=0
        for job in range(len(self.operation_num_per_jobs)):
            operation_num_per_jobs=self.operation_num_per_jobs[job]
            for i in range(operation_num_per_jobs):
                for k in range(j,j+self.operation_num_per_jobs[job]-1):
                    if individual.features[1][k]>individual.features[1][k+1]:
                        tmp = individual.features[1][k]
                        individual.features[1][k] = individual.features[1][k+1]
                        individual.features[1][k+1] = tmp
            j+=operation_num_per_jobs
        job=self.operation_num_per_jobs
        operation_index=0
        job_index=0
        for times in range(self.variables_range[1]): #for all operation
            #改到這
            if job[job_index]==operation_index:
                operation_index=0
                job_index+=1
            machine_index = individual.features[0][times]
            if self.job_machine_operation_map[job_index][operation_index][machine_index]==10000:
                N_machine = self.variables_range[0][0][1]+1
                for m in range(N_machine): #find the machine can operate this operation
                    if self.job_machine_operation_map[job_index][operation_index][m] != 10000:
                        individual.features[0][times] = m
                        break
            operation_index +=1
        return individual
    def generate_machine_schedule(self,job_nums):
        machine_nums = len(self.job_machine_operation_map[0][0])
        tmp = [] 
        for job_index in range(job_nums):
            operate_nums = self.variables_range[1][job_index]
            record_machine_index_can_operate = [[] for _ in range(operate_nums)]
            for operate in range(operate_nums):            
                for machine in range(machine_nums):
                    if self.job_machine_operation_map[job_index][operate][machine]<10000:
                        record_machine_index_can_operate[operate].append(machine)    
            tmp.append(record_machine_index_can_operate)
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
    def generate_individual_2(self):
        individual = myIndividual()
        individual.features.append([int(random.uniform(*x)) for x in self.variables_range[0]])
        tmp = np.array(range(self.variables_range[1]))
        random.shuffle(tmp)
        individual.features.append(tmp)
        individual = self.valid_individual(individual)
        #print("test")
        return individual
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
                for j in range(self.variables_range[1][i]):
                    schedule.append([i,2])
        machine_schedule.append(self.generate_machine_schedule(job_nums))
        random.shuffle(schedule)
        tmp_feature.append(schedule)
        tmp_feature.append(machine_schedule)
        individual.features.append(tmp_feature)
        #individual = self.valid_individual(individual)
        #print("test")
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
        half_population = int(self.num_of_individuals/2)
        while len(children) < half_population:
            children.append(mutate_population.fronts[0][0])
        # mutate half best individual
        for individual in children:
            cnt_mutate = 0
            #print(individual)
            operation_index = len(individual.features[0][-2])-1
            while cnt_mutate < num_mutate :
                index1 = random.randint(0,operation_index)
                index2 = random.randint(0,operation_index)
                if individual.features[0][-2][index1]!=individual.features[0][-2][index2]:
                    tmp = individual.features[0][-2][index1]
                    individual.features[0][-2][index1] = individual.features[0][-2][index2]
                    individual.features[0][-2][index2] = tmp
                cnt_mutate+=1
            
            mutate_flag = 0
            new = [self.problem.generate_machine_schedule(len(self.problem.variables_range[0]))]
            for batch in range(len(individual.features[0][-1])):
                for job in range(len(individual.features[0][-1][batch])):
                    for operation in range(len(individual.features[0][-1][batch][job])):
                        if individual.features[0][-1][batch][job][operation] != new[batch][job][operation]:
                            individual.features[0][-1][batch][job][operation] = new[batch][job][operation]
                            mutate_flag = 1
                            break
                    if mutate_flag == 1 :
                        break
        return children
    def __reseeding(self, population):
        half_population = int(self.num_of_individuals/2) # reseed half of initial population 
        children = Population()
        for _ in range(half_population):
            individual = self.problem.generate_individual()
            children.append(individual)
        return children
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
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        num_mutate = self.mutation_schedule[0][1]
        mutate_index = 1
        
        returned_population = None
        for i in range(self.num_of_generations):
            print('generation : ' + str(i))
            children = self.utils.create_children(self.population,num_mutate)
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            # for tmptest in self.population.fronts:
            #     print(len(tmptest))
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
            returned_population = self.population
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            if mutate_index < len(self.mutation_schedule) and i == self.mutation_schedule[mutate_index][0]:
                num_mutate = self.mutation_schedule[mutate_index][1]
                mutate_index+=1
        return returned_population.fronts[0]