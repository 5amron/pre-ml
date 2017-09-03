import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
import timeit
from . import heuristics
from ..io import portie
import sys







def baco(x_data, y_data, t_percent=40, heu_meth="method_1", ml_alg="knn1", iter_num=10):


    (my_bool, msg_err) = check_baco_args(t_percent, heu_meth, ml_alg, iter_num)
    if(not my_bool):
        print("problem with arguments for abaco()!!!")
        print(msg_err)
        exit() #############

    check = portie.CheckDataset(x_data, y_data)
    (state, msg) = check.get_state()
    if(not state): # data had problems
        print("+++ " + msg + " +++")
        exit() #############


    train_percentage = 100 - int(t_percent)

    time_temp = 0
    start = timeit.default_timer()
    (best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, best_fit_so_far, best_ant_road) = run_feature_selection(generations = iter_num, alpha = 1, beta = 0.5, T0 = 0.1, Min_T = 0.1, Max_T = 6, q = 0.95, Q = 0.3, heu_meth = heu_meth, ant_num = 50, feature_num = len(x_data[1]), dataset=x_data, targets=y_data, train_percentage=train_percentage)
    end = timeit.default_timer()
    time_temp = time_temp + (end - start)


    # making new dataset :
    new_dataset = make_new_dataset(best_ant_road, x_data)

    acc_before_run = get_single_fit(x_data, y_data, train_percentage)

    total_feature_num = len(x_data[1])
    sample_num = len(x_data[:,1])

    best_selected_features_num = np.sum(best_ant_road)


    return (new_dataset, best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num, best_fitnesses_each_iter, average_fitnesses_each_iter ,num_of_features_selected_by_best_ant_each_iter, time_temp, sample_num)











def check_baco_args(t_percent, heu_meth, ml_alg, iter_num):
    msg_err = ""
    try:
        int(t_percent)
    except Exception as e:
        msg_err = "t_percent should be integer!"
        return (False, msg_err)

    try:
        int(iter_num)
    except Exception as e:
        msg_err = "iter_num should be integer!"
        return (False, msg_err)

    if(iter_num > 100):
        msg_err = "iter_num should be less than 100!"
        return (False, msg_err)

    if(iter_num < 5):
        msg_err = "iter_num should be more than 5!"
        return (False, msg_err)

    # if(type(heu_meth) != "str" or "str" != type(ml_alg)):
    if(heu_meth != "method_1" and heu_meth != "method_2" and heu_meth != "method_3" and heu_meth != "method_4" and heu_meth != "method_5"):
        msg_err = "heu_meth isn't write, please check the docs!"
        return (False, msg_err)


    # should check the ml_alg tooooooooo

    return (True, msg_err)













def run_feature_selection(generations, alpha, beta , T0, Min_T, Max_T, q, Q, heu_meth, ant_num, feature_num, dataset, targets, train_percentage):

    best_fitnesses_each_iter = []
    average_fitnesses_each_iter = []
    num_of_features_selected_by_best_ant_each_iter = []
    road_map = np.random.randint(2, size=ant_num*feature_num).reshape((ant_num, feature_num))
    road_maps = np.zeros(ant_num*feature_num*generations, dtype="int64").reshape(generations, ant_num, feature_num)
    best_roads_list = []

    best_fit_so_far = 0
    best_road_so_far = np.zeros(feature_num, dtype="int64")

    np.set_printoptions(suppress=True, threshold=1000)

    roads_T = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num) + T0


    for i in range(0, generations):
        # print("+++++++++ run : ("+ str(heu_meth) +") Iteration : (" + str(i+1) + ")+++++++++")

        if(heu_meth == "method_1"):
            roads_E = heuristics.hueristic_value_fscore(feature_num, dataset, targets)
        elif(heu_meth == "method_2"):
            roads_E = heuristics.heuristic_value_min_redundency(feature_num, dataset)
        elif(heu_meth == "method_3"):
            roads_E = heuristics.heuristic_value_min_redundency_max_relevence(feature_num, dataset)
        elif(heu_meth == "method_4"):
            roads_E = heuristics.heuristic_value_method_4(feature_num, dataset)
        elif(heu_meth == "method_5"):
            roads_E = heuristics.heuristic_value_mutual_info(feature_num, dataset)


        (road_map, pointer) = baco_road_selection(roads_T, roads_E, alpha, beta, ant_num, feature_num)

        (iter_best_fit, best_road_so_far, best_fit_so_far, iter_best_road, fitnesses, iter_average_fit, ants_num_of_features_selected) = do_calculations(road_map, dataset, targets, best_fit_so_far, best_road_so_far, train_percentage)

        roads_T = trial_update(fitnesses, roads_T, Min_T, Max_T, Q, q, iter_best_road, feature_num)

        road_maps[i] = road_map
        best_fitnesses_each_iter.append(iter_best_fit)
        average_fitnesses_each_iter.append(iter_average_fit)
        num_of_features_selected_by_best_ant_each_iter.append(sum(best_road_so_far))
        best_roads_list.append(best_road_so_far)


    ccc = 0
    maxx = max(best_fitnesses_each_iter)
    for each in best_fitnesses_each_iter:
        if(each == maxx):
            my_indx = ccc
        ccc = ccc + 1

    return (best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, best_fit_so_far, best_roads_list[my_indx])
















def get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets):
    K = 1
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(train_dataset, train_targets) # X, Y

    # evaluating our trained model
    predicted_targets = knn.predict(test_dataset)

    l = len(test_targets)
    num_of_correct = 0
    for i in range(l):
        if(test_targets[i] == predicted_targets[i]):
            num_of_correct = num_of_correct + 1
    return num_of_correct/l






def separate_datasets(dataset, targets, train_percentage):

    # in case you wanted the data to be random every single time you wanted get fitnesses
    leng = len(dataset[:, 0])
    s = int(leng*(train_percentage/100))

    samples_list = random.sample(range(0, leng), s)

    mask = np.zeros((leng), dtype=bool)
    mask[samples_list] = True

    train_dataset = dataset[mask, :]
    test_dataset = dataset[~mask, :]

    train_targets = targets[mask]
    test_targets = targets[~mask]


    return (train_dataset, test_dataset, train_targets, test_targets)








def get_fitnesses(road_map, dataset, targets, train_percentage):
    total_feature_num = len(road_map[1])
    total_sample_num = len(dataset[:,0])
    num_of_features_selected = list()
    fitnesses = list()

    count = 0
    for ant_solution in road_map:
        count = count + 1
        if np.sum(ant_solution) == 0:
            print("all of row "+ str(count) +" was 0!!!")
            fitnesses.append(0)
             # print(np.sum(ant_solution)) ##### problemmmmmm
        else:
            new_dataset = np.zeros(total_sample_num, dtype="float64").reshape(total_sample_num, 1)

            for i in range(0, len(ant_solution)):
                if(ant_solution[i] == 1):
                    new_dataset = np.append(new_dataset, dataset[:, i].reshape(total_sample_num, 1), axis=1)

            new_dataset = np.delete(new_dataset, 0, axis=1) # removing first column

            num_of_features_selected.append(new_dataset.shape[1])

            (train_dataset, test_dataset, train_targets, test_targets) = separate_datasets(new_dataset, targets, train_percentage)

            fitnesses.append(get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets))

    return num_of_features_selected, fitnesses








def make_new_dataset(solution_road, dataset):
    total_sample_num = len(dataset[:,0])
    new_dataset = np.zeros(total_sample_num, dtype="float64").reshape(total_sample_num, 1)

    if np.sum(solution_road) == 0:
        print("allll of it was 0!!!!")
        return new_dataset
    else:
        for i in range(0, len(solution_road)):
            if(solution_road[i] == 1):
                new_dataset = np.append(new_dataset, dataset[:, i].reshape(total_sample_num, 1), axis=1)

        new_dataset = np.delete(new_dataset, 0, axis=1) # removing first column
        return new_dataset








def get_single_fit(dataset, targets, train_percentage):

    (train_dataset, test_dataset, train_targets, test_targets) = separate_datasets(dataset, targets, train_percentage)

    return get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets)



def roulette_wheel(probs, feature_num): # picking one item's index randomly but with considering probabilities.
    sum = 0
    zero_or_one = 1
    r = np.random.random_sample()
    for x in range(len(probs)):
        sum = sum + probs[x]
        if(r < sum):
            index = x
            # because it is now (feature_num + feature_num) long, we should correct it :
            if(index >= feature_num):
                index = index - feature_num
                zero_or_one = 1
            else:
                zero_or_one = 0
            return (index, zero_or_one)






def baco_road_selection(roads_T, roads_E, alpha, beta, ant_num, feature_num):
    road_map = np.zeros(ant_num*feature_num, dtype="int64").reshape(ant_num, feature_num)
    pointer = np.zeros(ant_num*feature_num, dtype="int64").reshape(ant_num, feature_num)

    for k in range(0, ant_num):
        indx = np.multiply(np.power(roads_T, alpha), np.power(roads_E, beta))
        for j in range(0, feature_num):

            # for the first feature :
            if(j == 0):
                cur_feature = np.random.randint(0, feature_num, 1)[0]
                pointer[k,j] = cur_feature
                # this is just for selection of 0 or 1 for the first feature (if it's more interesting the likelihood is higher)
                temp = np.sum(roads_T[0, :, cur_feature] + roads_T[2, :, cur_feature]) / np.sum(roads_T[0, :, cur_feature] + roads_T[1, :, cur_feature] + roads_T[2, :, cur_feature] + roads_T[3, :, cur_feature])
                rand = np.random.random_sample()

                if (rand < temp):
                    road_map[k, cur_feature] = 0
                else:
                    road_map[k, cur_feature] = 1

            else:
                if(road_map[k, pointer[k,j-1]] == 1):
                    nominator = np.hstack((indx[2, pointer[k,j-1], :], indx[3, pointer[k,j-1], :]))
                    denominator = sum(nominator) ##################################### should be right!!!!!
                    probability = np.divide(nominator, denominator) # total=total/sum(total) # should be editted.it is not
                    (selected_feature_indx, zero_or_one) = roulette_wheel(probability, feature_num)
                    pointer[k,j] = selected_feature_indx


                    if(zero_or_one == 0):
                        road_map[k, pointer[k,j]] = 0
                    else:
                        road_map[k, pointer[k,j]] = 1

                else: # == 0
                    nominator = np.hstack((indx[0, pointer[k,j-1], :], indx[1, pointer[k,j-1], :]))
                    denominator = sum(nominator)
                    probability = np.divide(nominator, denominator)
                    (selected_feature_indx, zero_or_one) = roulette_wheel(probability, feature_num)
                    pointer[k,j] = selected_feature_indx


                    if(zero_or_one == 0):
                        road_map[k, pointer[k,j]] = 0
                    else:
                        road_map[k, pointer[k,j]] = 1

            # update indx (so by doing this, the probability of selection in roulette wheel for this feature, is gonna be zero!)
            indx[:, :, pointer[k, j]] = 0


    return (road_map, pointer)








def do_calculations(road_map, dataset, targets, best_fit_so_far, best_road_so_far, train_percentage):

    ants_num_of_features_selected, fitnesses = get_fitnesses(road_map, dataset, targets, train_percentage)

    iter_average_fit = np.mean(fitnesses, axis=0)
    iter_best_fit = max(fitnesses)
    iter_best_ant = fitnesses.index(iter_best_fit)
    iter_best_road = road_map[iter_best_ant, :]

    if(iter_best_fit > best_fit_so_far):
        best_fit_so_far = iter_best_fit
        best_road_so_far = iter_best_road


    return (iter_best_fit, best_road_so_far, best_fit_so_far, iter_best_road, fitnesses, iter_average_fit, ants_num_of_features_selected)








def trial_update(fitnesses, roads_T, Min_T, Max_T, Q, q, iter_best_road, feature_num):

    roads_T = roads_T * q # i think this is pheromone evaporation
    # class_err = 1 - fitnesses # not this because fitnesses is a list and doesn't work this way
    class_err = np.array([1-i for i in fitnesses])
    min_err = min(class_err)
    min_err_indx = np.where(class_err == min_err)[0][0]


    roads_T_temp = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)

    # here we assign one to best road edges in roads_T_temp.
    for i in range(0, len(iter_best_road)):
        if(iter_best_road[i] == 0):
            roads_T_temp[0, :, i] = 1
            roads_T_temp[2, :, i] = 1
        else:
            roads_T_temp[1, :, i] = 1
            roads_T_temp[3, :, i] = 1


    if(class_err[min_err_indx] == 0):
        roads_T_temp = (Q/(class_err[min_err_indx] + 0.001)) * roads_T_temp
    else:
        roads_T_temp = (Q/(class_err[min_err_indx])) * roads_T_temp

    roads_T = roads_T + roads_T_temp
    # now we make sure all of them are in interval :
    for each in np.nditer(roads_T, op_flags=['readwrite']):
        if(each > Max_T):
            each[...] = Max_T
        else:
            if(each < Min_T):
                each[...] = Min_T

#     print(roads_T)
#     roads_T = np.add(roads_T[roads_T < Min_T] * Min_T , roads_T * roads_T[roads_T > Min_T]) +++++++++++!
#     roads_T = np.add(roads_T[roads_T > Max_T] * Max_T , roads_T * roads_T[roads_T < Max_T]) +++++++++++!


    return roads_T















def get_Fr(feature_num, selected):
    return (feature_num-selected)/feature_num
