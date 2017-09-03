import numpy as np
import sklearn







def hueristic_value_fscore(feature_num, dataset, targets):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)

#     arr = np.corrcoef(dataset)
#     R = abs(arr)

    ## F-score :
    classes = np.unique(targets)
    class_num = len(classes)
    total_mean_f = dataset.mean(0)
    nominator = 0
    denominator = 0

#     nominator = np.zeros(feature_num, dtype="int64")
#     denominator = np.zeros(feature_num, dtype="int64")

    sample_num_of_this_tag = np.zeros(class_num, dtype="int64")
    for i in range(0, class_num):
        tags = np.zeros((len(targets)), dtype="int64")
        bool_arr = np.equal(targets, classes[i])
        tags[bool_arr] = 1
        sample_num_of_this_tag[i] = np.sum(tags)
        dataset_only_class = dataset[bool_arr, :]
        class_mean_f = dataset_only_class.mean(0)
        class_mean_f = np.round(class_mean_f, decimals=4)


        nominator = nominator + np.power(np.subtract(class_mean_f, total_mean_f), 2)
        denominator = denominator + sum(np.power(np.subtract(dataset_only_class, np.matlib.repmat(total_mean_f, dataset_only_class.shape[0],1)), 2)) / (sample_num_of_this_tag[i]-1)

    F_score = np.divide(nominator, denominator)


    roads_E[0, :, :] = (0.5/feature_num) * sum(F_score)
    roads_E[1, :, :] = np.matlib.repmat(F_score, feature_num, 1)

    roads_E[2, :, :] = (0.5/feature_num) * sum(F_score)
    roads_E[3, :, :] = np.matlib.repmat(F_score, feature_num, 1)

    return roads_E


















def heuristic_value_min_redundency(feature_num, dataset):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    arr = np.corrcoef(dataset.T)
    R = abs(arr)

    roads_E[0, :, :] = R
    roads_E[1, :, :] = 1 - R

    roads_E[2, :, :] = R
    roads_E[3, :, :] = 1 - R
    return roads_E
















def heuristic_value_min_redundency_max_relevence(feature_num, dataset):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    class_corrs = np.zeros(feature_num, dtype="float64")
    arr = np.corrcoef(dataset.T)
    R = abs(arr)

    for i in range(0, feature_num):
        class_corrs[i] = abs(np.corrcoef(dataset.T[i, :], targets)[0, 1])

    class_corr_sel = np.matlib.repmat(class_corrs, feature_num, 1)
    class_corr_desel = np.matlib.repmat(1-class_corrs, feature_num, 1)

    roads_E[0, :, :] = np.sqrt(np.multiply(R, class_corr_desel))
    roads_E[1, :, :] = np.sqrt(np.multiply(1-R, class_corr_sel))

    roads_E[2, :, :] = np.sqrt(np.multiply(R, class_corr_desel))
    roads_E[3, :, :] = np.sqrt(np.multiply(1-R, class_corr_sel))
    return roads_E















def heuristic_value_method_4(feature_num, dataset):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    class_corrs = np.zeros(feature_num, dtype="float64")
    arr = np.corrcoef(dataset.T)
    R = abs(arr)

    # has problem (i don know MI and it's getting values bigger than 1 ...)
    for i in range(0, feature_num):
        class_corrs[i] = abs(np.corrcoef(dataset.T[i, :], targets)[0, 1])

    class_corr_sel = np.matlib.repmat(class_corrs, feature_num, 1)
    class_corr_desel = np.matlib.repmat(1-class_corrs, feature_num, 1)

    roads_E[0, :, :] = class_corr_desel
    roads_E[1, :, :] = class_corr_sel

    roads_E[2, :, :] = R
    roads_E[3, :, :] = 1 - R

    return roads_E










def heuristic_value_mutual_info(feature_num, dataset):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    mutual_f_f = np.zeros(feature_num*feature_num, dtype="float64").reshape(feature_num, feature_num)


    for i in range(0, feature_num):
        for j in range(0, feature_num):
            if(i == j):
                mutual_f_f[i, j] = 1
            else:
                mutual_f_f[i, j] = sklearn.metrics.normalized_mutual_info_score(dataset[:, i], dataset[:, j])

#     print(mutual_f_f)
    roads_E[0, :, :] = mutual_f_f
    roads_E[1, :, :] = 1 - mutual_f_f

    roads_E[2, :, :] = mutual_f_f
    roads_E[3, :, :] = 1 - mutual_f_f

    return roads_E
