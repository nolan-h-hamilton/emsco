import random
import math
import numpy.random
import scipy.stats


class Chromosome:

    def __init__(self):
        self.stage_list = []
        self.accuracy = None 
        self.conclusiveness = 0.0 
        self.time = 0.0           
        self.inv_time = 0.0       
        self.rank = 0.0           
        self.fitness = 0.0        
        self.hash = None          


    def __eq__(self, other):
        if isinstance(other, Chromosome):
            
            return self.stage_list == other.stage_list and self.rank == other.rank
        return False


    def __repr__(self):
        return ('(acc (g2): %5.3f, coverage (g1): %5.3f, cost (g3*):%5.2f), [Q]: %s'
                %(self.accuracy, self.conclusiveness, self.time,
                 self.stage_list))
                 

    def set_fitness(self,alpha,scale=100):
        alpha = alpha
        weight = 1.0
        scale=1
        self.fitness = math.sqrt(pow(self.accuracy*scale*weight, 2) + \
            pow(self.conclusiveness*scale, 2) + \
            pow(self.inv_time*scale, 2))*pow(alpha, self.rank)

                
    def set_hash(self):
        assert self.stage_list
        stages_str = ''.join(str(stage) for stage in self.stage_list)
        self.hash = stages_str

            
    def mutate(self, mutation_prob, max_stages, a=1, b=2):
        
        for index,_ in enumerate(self.stage_list):
            stage_index_limit = max(self.stage_list)
            if random.uniform(0,1) < mutation_prob:
                stage_index_limit = min(stage_index_limit + 1, (max_stages - 1))

                # beta binomial distribution with a=1, b>1 has monotonically
                # decreasing probability and p(x > stage_index_limit) = 0
                distr = scipy.stats.betabinom(stage_index_limit, a, b)
                draw = int(numpy.random.choice(range(stage_index_limit+1), 1,
                              [distr.pmf(x) for x in range(stage_index_limit)]))
                
                self.stage_list[index] = draw
                
        self.realign_stage_list()

                
    def consecutive_stages(self):
        return list(set(self.stage_list)) == list(range(0, max(self.stage_list) + 1))


    def realign_stage_list(self):
        if not self.consecutive_stages():
            current_stage = 0
            for stage in set(self.stage_list):
                for index, value in enumerate(self.stage_list):
                    if value == stage:
                        self.stage_list[index] = current_stage
                current_stage += 1

        
