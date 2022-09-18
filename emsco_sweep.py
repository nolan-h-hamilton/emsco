"""Measure EMSCO Performance Across a Range of \hat{p} Values"""
import argparse
import math
import random
import time
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from chromosome import Chromosome


def str_rep(chrom):
    """returns comma-separated string representation of chromosome"""
    srep = ''
    for elem in chrom.stage_list:
        srep += str(elem) + ','
    return srep[:-1]


def l2_norm(acc, conc, inv_time):
    """returns \mathscr{E}(\cdot)"""
    return math.sqrt((acc**2) + (conc**2) + (inv_time**2))


def gamma(population, eps=.01):
    """compute \gamma for a given generation/population during iteration"""
    temp_population = deepcopy(population)
    temp_population.sort(key=lambda x: x.rank, reverse=True)
    rnk_lst = list(set([item.rank for item in temp_population]))
    rnk_lst.sort(reverse=True)
    sum_lst = [l2_norm(item.accuracy, item.conclusiveness, item.inv_time)
                  for item in temp_population]
    u_E = max(sum_lst)
    l_E = min(sum_lst)
    return (u_E / l_E) + eps


def rank(population):
    """rank chromosomes in `population' according to non-domination level"""
    for index, chromosome in enumerate(population):
        population[index].fitness = 0.0
        population[index].rank = 0.0
        population[index].inv_time = 0.0
    min_time = min([chromosome.time for chromosome in population])
    for index, _ in enumerate(population):
        population[index].inv_time = round(min_time / population[index].time, 3)
    assert min([chromosome.inv_time for chromsome in population]) > 0.0
    current_rank = 1
    temp_population = deepcopy(population)
    while temp_population != []:
        temp_population, top_chromosomes = get_top_rank(temp_population)
        for index, chromosome in enumerate(top_chromosomes):
            population[population.index(chromosome)].rank = current_rank
        current_rank += 1
    max_rank = current_rank
    for chromosome in population:
        max_rank = max(max_rank, chromosome.rank)
    for index, chromosome in enumerate(population):
        population[index].rank = (max_rank + 1) - population[index].rank
    opt_gamma = gamma(population)
    for index, chromosome in enumerate(population):
        population[index].set_fitness(float(opt_gamma))


def get_top_rank(population):
    """return non-dominated solutions in `population'"""
    rem_lst = population[1:]   # unranked list
    top_lst = [population[0]]  # ranked list (Pareto front)
    i = 0
    while i < len(rem_lst):
        dominated = False
        dom_lst = []
        for top in top_lst:
            if (rem_lst[i].accuracy >= top.accuracy and
                    rem_lst[i].conclusiveness >= top.conclusiveness
                    and rem_lst[i].inv_time >= top.inv_time) and \
                    (rem_lst[i].accuracy > top.accuracy or
                    rem_lst[i].conclusiveness > top.conclusiveness or
                    rem_lst[i].inv_time > top.inv_time):
                assert not dominated
                dom_lst.append(top)

            elif (top.accuracy >= rem_lst[i].accuracy and
                     top.conclusiveness >= rem_lst[i].conclusiveness and
                     top.inv_time >= rem_lst[i].inv_time) and \
                     (top.accuracy > rem_lst[i].accuracy or
                     top.conclusiveness > rem_lst[i].conclusiveness or
                     top.inv_time > rem_lst[i].inv_time):
                dominated = True

        if not dominated:
            for chromosome in dom_lst:
                top_lst.remove(chromosome)
                rem_lst.append(chromosome)
            top_lst.append(rem_lst[i])
            rem_lst.remove(rem_lst[i])
            i = 0
        else:
            i += 1
    return deepcopy(rem_lst), deepcopy(top_lst)


def select_chromosome(population):
    """execute fitness-proportionate solution"""
    total_fitness = sum([chromosome.fitness for chromosome in population])
    rand_number = randint(0, math.ceil(total_fitness))
    sum_fitness = 0
    for index, _ in enumerate(population):
        if rand_number <= sum_fitness:
            return population[index]
        sum_fitness += population[index].fitness
    return population[-1]


def crossover(r_hat, parent_a, parent_b):
    """perform crossover on `parent_a', `parent_b' to create new chromosome `child'"""
    child = deepcopy(parent_a)
    choices = [max(parent_a.stage_list), max(parent_b.stage_list), math.floor(
        (max(parent_a.stage_list) + max(parent_b.stage_list)) / 2)]
    k_child = random.choice(choices)
    if random.uniform(0, 1) < r_hat:
        for index, assn in enumerate(child.stage_list):
            parent_r = None
            flip = random.randint(0, 1)
            if flip == 0:
                parent_r = deepcopy(parent_a)
            else:
                parent_r = deepcopy(parent_b)
            k_pr = max(parent_r.stage_list)
            if k_pr != 0:
                child.stage_list[index] = round((parent_r.stage_list[index] / k_pr) * k_child)
            else:
                child.stage_list[index] = 0
            child.realign_stage_list()
    return child


def classify_dataframe(clf, train_df, test_df, features,
                       correct_label, min_proba,
                       solver_="sag",lr_iter=1000):
    """selectively classify inputs in test_df"""
    clf = LogisticRegression(max_iter=lr_iter,solver=solver_)
    clf_trained = clf.fit(train_df[features], train_df[correct_label])
    y_pred = clf_trained.predict(test_df[features])
    proba_table = clf_trained.predict_proba(test_df[features])
    classes = clf_trained.classes_.tolist()

    proba = np.empty([len(proba_table), 1])
    for index, row in enumerate(proba_table):
        proba[index, 0] = proba_table[index, classes.index(y_pred[index])]

    test_df['pred_label'] = np.asarray(y_pred)
    test_df['proba'] = np.asarray(proba)
    num_conclusive = 0
    for value in (test_df['proba'] >= min_proba):
        if value is True:
            num_conclusive += 1

    num_conclusive_incorrect = (test_df.loc[test_df['proba'] >= min_proba, correct_label]
        != test_df.loc[test_df['proba'] >= min_proba, 'pred_label']).tolist().count(True)

    inconclusive = test_df['proba'] < min_proba
    inconclusive_df = test_df[inconclusive].copy()
    return num_conclusive, num_conclusive_incorrect, inconclusive_df


def classify_with_stages(clf, train_df, test_df, stages,
                         feature_times, min_proba, index=-1,solver_="sag",lr_iter=1000):
    """run instances through sequential classification protocol"""
    feature_names = list(train_df.columns.values)[:-1]
    total_conclusive = 0
    total_conclusive_incorrect = 0
    total_time = 0
    inconclusive_df = test_df

    assert len(stages) == len(feature_names)

    stage_sorted_set = list(set(stages))
    stage_sorted_set.sort()

    for stage in stage_sorted_set:
        feature_indices = [index for index,
            feature in enumerate(stages) if feature <= stage]
        feature_lst = [feature_names[index] for index in feature_indices]

        num_conclusive, num_conclusive_incorrect, inconclusive_df = classify_dataframe(clf,
            train_df, inconclusive_df, feature_lst, 'label', min_proba, solver_= solver_,
            lr_iter=lr_iter)
        total_conclusive += num_conclusive
        total_conclusive_incorrect += num_conclusive_incorrect

        feature_time_indices = [index for index, feature in enumerate(stages)
            if feature <= stage]

        total_time += sum([feature_times[index] for index in feature_time_indices]) \
                         * num_conclusive

        if len(inconclusive_df) == 0 and stage < max(stage_sorted_set):
            break

    if len(inconclusive_df) > 0:
        assert stage == max(stage_sorted_set)
        total_time += sum([feature_times[index] for index in feature_time_indices]) \
                         * len(inconclusive_df)

    final_accuracy = (total_conclusive -
                      total_conclusive_incorrect) / total_conclusive
    final_inconclusive_percentage = len(
        inconclusive_df.index) / len(test_df.index)

    return index, final_accuracy, 1 - \
        final_inconclusive_percentage, total_time / len(test_df)


def random_chromosome(clf, train_df, test_df, num_feature,
                      feature_times, min_proba, max_stages, solver_="sag",
                      lr_iter=1000, stage_list=None, mut_prob=0.5):
    """
    generate a random 2-stage chromosome (solution and score)
    """
    if stage_list is None:
        stage_list = [0 for i in range(len(feature_times))]
    chrom = Chromosome()
    chrom.stage_list = stage_list
    chrom.realign_stage_list()
    chrom.mutate(mut_prob, 2)
    chrom.realign_stage_list()
    chrom.set_hash()

    index, chrom.accuracy, chrom.conclusiveness, chrom.time = classify_with_stages(clf,
        train_df.copy(), test_df.copy(), stage_list, feature_times, min_proba,
        solver_=solver_,lr_iter=lr_iter)
    return chrom


def sort_population(population):
    """
    sort the population (descending order) based on chromosome fitness
    :population is a list of chromosomes
    """
    return sorted(population, key=lambda chromosome: chromosome.fitness,
                  reverse=True)


def chromosome_hash(chromosome):
    """
    create unique identifier for a given stage_list
    :chromosome is a solution and score
    """
    num = 1
    hash_sum = 0
    for sub_list in chromosome.stage_list:
        hash_sum += hash(frozenset(sub_list)) * num
        num *= 2
    return hash_sum


def find_best_chromosome(population):
    """return copy of 'best' chromosome in population"""
    best_fitness = population[0].fitness
    best_lst = [chromosome for chromosome in population if chromosome.fitness == best_fitness]
    best_index = 0
    for index, _ in enumerate(best_lst):
        if max(best_lst[index].stage_list) < max(
            best_lst[best_index].stage_list):
            best_index = index
    return deepcopy(population[best_index])


def main():
    date_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
    print(date_stamp)
    chromosome_dct = {}  # dictionary of unique chromosomes

    parser = argparse.ArgumentParser(
        description='Evolutionary Multi-stage Classifier Optimizer -- Sweep Confidence Threshold')
    parser.add_argument('--train_data', help='CSV file containing training data')
    parser.add_argument('--test_data', help='CSV file containing test data')
    parser.add_argument('--val_data')
    parser.add_argument('--cost_data',
                        help='single row of comma-separated cost vals')
    parser.add_argument('--pop_size', type=int, default=300,
                        help='number of chromosomes in each population/generation')
    parser.add_argument('--iter_num', type=int, default=150,
                        help='number of generations to evaluate before termination')
    parser.add_argument('--elite', type=float, default=0.2,
                        help='fraction of each generation that is copied to next')
    parser.add_argument('--crossover_rate', type=float, default=0.8)
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='probability of an index undergoing mutation')
    parser.add_argument('--min_prob', type=float, default=0.50,
                        help='prediction confidence threshold (p*)')
    parser.add_argument('--max_stages', type=int)
    parser.add_argument('--inc', type=int, default=1)
    parser.add_argument('--bias', type=float, default=2,
                        help="beta parameter affecting mutation")
    parser.add_argument('--output_length', type=int, default=10)
    parser.add_argument('--sweep',type=float, default=.10)
    parser.add_argument('--exp_num', type=int, default=5)
    parser.add_argument('--lr_iter', type=int, default=1000)
    parser.add_argument('--lr_solver',type=str, default="sag",
        help="for deterministic gradient descent, use `lbfgs` instead")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--out',type=str, default = "sweep_output_{}.txt".format(date_stamp))
    args = vars(parser.parse_args())

    max_elite_size = args['pop_size']
    iter_num = args['iter_num']
    elite_percentage = args['elite']
    crossover_rate = args['crossover_rate']
    mutation_rate = args['mutation_rate']
    min_proba = args['min_prob']
    max_stages = args['max_stages']
    population_size = args['pop_size']
    feature_times = [float(x) for x in open(args['cost_data'], 'r').read().strip().split(',')]
    train_df = pd.read_csv(args['train_data'])
    test_df = pd.read_csv(args['test_data'])
    val_df = pd.read_csv(args['val_data'])
    val_columns = ['f' + str(i) for i in range(len(feature_times))]
    train_columns = ['f' + str(i) for i in range(len(feature_times))]
    test_columns = ['f' + str(i) for i in range(len(feature_times))]
    train_columns.append('label')
    test_columns.append('label')
    val_columns.append('label')
    val_df.columns = val_columns
    train_df.columns = train_columns
    test_df.columns = test_columns
    clf = LogisticRegression(max_iter=args['lr_iter'],solver=args['lr_solver'])
    num_feature = len(feature_times)
    phat_best = {}
    assert max_stages <= num_feature
    init_time = time.time()

    for exp in range(args['exp_num']):
        if exp > 0:
            # increment confidence threshold
            min_proba += args['sweep']
            min_proba = round(min_proba,3)
        phat_best.update({min_proba:[]})
        for run in range(args['runs']):
            pool = mp.Pool()
            print('\nEXP: ({}/{}), RUN: ({}/{})'.format(exp+1,args['exp_num'], run+1, args['runs']))

            results = [pool.apply_async(random_chromosome, args=(clf, train_df.copy(), val_df.copy(),
            num_feature, feature_times, min_proba, max_stages,args['lr_solver'],args['lr_iter']))
            for index in range(population_size)]
            
            pool.close()
            pool.join()
            population = []
            for res in results:
                population.append(res.get())
            rank(population)
            initial_population = sort_population(population)
            population = deepcopy(initial_population)
            for chromosome in initial_population:
                chromosome_dct.update({chromosome.hash: deepcopy(chromosome)})
            # begin iteration
            for i in range(iter_num):
                print("\ngeneration {}, time_elapsed (minutes): {}".format(
                    i,round((time.time()-init_time)/60,3)))
                nd_front = len(get_top_rank(population)[1])
                e0 = max(nd_front, math.floor(
                    len(population) * elite_percentage))
                elite_population = population[:e0]
                # remove redundant solutions from elite population
                new = []
                seen = []
                for es in elite_population:
                    if es.stage_list in seen:
                        continue
                    seen.append(es.stage_list)
                    new.append(es)
                elite_population = new
                # population size increment/decrement
                if len(elite_population) >= max_elite_size:
                    population_size += args['inc']
                if population_size > args['pop_size'] and len(elite_population) < population_size-args['inc']:
                    population_size -= args['inc']
                # per-generation validation performance output
                print("confidence threshold: {}".format(min_proba))
                print("size of elite population: {}".format(len(elite_population)))
                print("validation performance -- top-ranked solutions")
                for sol in elite_population[0:args['output_length']]:
                    print(sol)

                new_population = deepcopy(elite_population)
                while len(new_population) < population_size:
                    mate1 = select_chromosome(population)
                    mate2 = select_chromosome(population)
                    new_chromosome = crossover(crossover_rate, mate1, mate2)
                    new_chromosome.mutate(mutation_rate, max_stages,1,args['bias'])
                    new_chromosome.set_hash()
                    new_chromosome.rank = 0
                    new_chromosome.fitness = 0.0
                    new_population.append(new_chromosome)
                assert len(new_population) == population_size

                output = mp.Queue()
                pool = mp.Pool()
                result = []
                for index in range(population_size):
                    if new_population[index].hash in chromosome_dct:
                        hsh = new_population[index].hash
                        new_population[index].accuracy = chromosome_dct[hsh].accuracy
                        new_population[index].conclusiveness = chromosome_dct[hsh].conclusiveness
                        new_population[index].time = chromosome_dct[hsh].time
                    else:
                        result.append(pool.apply_async(classify_with_stages,
                            args=(clf, train_df.copy(), val_df.copy(),new_population[index].stage_list,
                            feature_times, min_proba, index, args['lr_solver'], args['lr_iter'])))
                pool.close()
                pool.join()

                for res in result:
                    index, new_population[index].accuracy, new_population[index].conclusiveness, new_population[index].time = res.get()
                rank(new_population)
                new_population = sort_population(new_population)
                population = new_population
                for c_,chrom_ in enumerate(new_population):
                    chromosome_dct.update({new_population[c_].hash: deepcopy(new_population[c_])})

                if i == args['iter_num']-1:
                    """
                    store the top-ranked solution for the current run's
                    terminal generation. Performance measured w/ validation set
                    """
                    phat_best[min_proba].append(elite_population[0])

                if run == args['runs']-1 and i == args['iter_num']-1:
                    """
                    now take each list in `phat_best` and compute
                    the test performance for each solution. average
                    the results to estimate for EMSCO's out-of-sample
                    performance. results for each \hat{p} stored in 
                    a tab-separated file given by `args['out']`.
                    """
                    print('\ncomputing test performance and writing output to {}'.format(args['out']))
                    acc_avg = cov_avg = cost_avg = acc_std = cov_std = cost_std = 0
                    for sol in phat_best[min_proba]:
                        perf = classify_with_stages(clf, train_df, test_df,
                          sol.stage_list, feature_times, min_proba, solver_=args['lr_solver'],
                          lr_iter=args['lr_iter'])
                        acc_avg += perf[1]
                        acc_std += perf[1]**2

                        cov_avg += perf[2]
                        cov_std += perf[2]**2

                        cost_avg += perf[3]
                        cost_std += perf[3]**2
                        
                    acc_avg /= args['runs']
                    cov_avg /= args['runs']
                    cost_avg /= args['runs']
                    acc_std  = math.sqrt(acc_std/(args['runs']) - (acc_avg**2))
                    cov_std = math.sqrt(cov_std/(args['runs']) - (cov_avg**2))
                    cost_std = math.sqrt(cost_std/(args['runs']) - (cost_avg**2))
                    output_file = open(args['out'], 'a+', encoding="utf-8")
                    if exp == 0:
                        output_file.write("runs\tp_hat\ttest_accuracy\ttest_coverage\ttest_cost\tacc_std\tcov_std\tcost_std\n")
                    
                    output_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        args['runs'], min_proba, round(acc_avg,3),
                        round(cov_avg,3), round(cost_avg,3),
                        round(acc_std,3), round(cov_std,3),
                        round(cost_std,3)) + "\n")
                    output_file.close()

if __name__ == '__main__':
    main()
