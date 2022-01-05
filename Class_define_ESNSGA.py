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
        tmp_feature = []
        for i in range(len(self.variables_range[0])):
            threshold = random.random()
            if self.variables_range[0][i] <2:
                tmp_feature.append(1)
            else:
                if threshold >0.5:
                    tmp_feature.append(1)
                else:
                    tmp_feature.append(2)
        schedule =[]
        for i in range(len(self.variables_range[0])):
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
        random.shuffle(schedule)
        tmp_feature.append(schedule)
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
    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                # print("equil_check")
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            
            children.append(child1)
            children.append(child2)

        return children
    def __mutate(self, child): 
        for gene_gene in range(len(child.features[0])):
            u, delta = self.__get_delta()
            if u < 0.5:
                child.features[0][gene_gene] = random.randint(self.problem.variables_range[0][gene_gene][0],self.problem.variables_range[0][gene_gene][1])
        child = self.problem.valid_individual(child)
    def __crossover(self, individual1, individual2):
        crossover_point = 2
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        tmp1 = []
        tmp2 = []
        for gene in range(len(individual1.features[0])):
            if gene < crossover_point:
                child1.features[0][gene] = individual1.features[0][gene]
                child2.features[0][gene] = individual2.features[0][gene]
            else:
                child1.features[0][gene] = individual2.features[0][gene]
                child2.features[0][gene] = individual2.features[0][gene]
            if individual1.features[1][gene] > crossover_point:
                tmp1.append(individual1.features[1][gene])
            if individual2.features[1][gene] > crossover_point:
                tmp2.append(individual2.features[1][gene])
        cnt=0
        for t in tmp1:
            while individual2.features[1][cnt] <= crossover_point:
                child2.features[1][cnt] = individual2.features[1][cnt]
                cnt+=1
            child2.features[1][cnt] = t
            cnt+=1
        cnt=0
        for t in tmp2:
            while individual1.features[1][cnt] <= crossover_point:
                child1.features[1][cnt] = individual1.features[1][cnt]
                cnt+=1
            child1.features[1][cnt] = t
            cnt+=1
        child1 = self.problem.valid_individual(child1)
        child2 = self.problem.valid_individual(child2)
        return child1, child2
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

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.utils = myUtils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
    def evolve(self):
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        for i in range(self.num_of_generations):
            print('generation : ' + str(i))
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
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
            children = self.utils.create_children(self.population)
        return returned_population.fronts[0]