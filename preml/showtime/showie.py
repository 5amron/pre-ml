from . import baco_show




# solution === (new_dataset, best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num, best_fitnesses_each_iter, average_fitnesses_each_iter ,num_of_features_selected_by_best_ant_each_iter, time_temp, sample_num)

def draw_baco(solution):

    if(len(solution) != 11):
        print("+++ can't draw the solution due to problem with it! +++")
        return

    (new_dataset, best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num, best_fitnesses_each_iter, average_fitnesses_each_iter ,num_of_features_selected_by_best_ant_each_iter, time_temp, sample_num) = solution

    baco_show.show_res_for_this_run(best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, total_feature_num)
