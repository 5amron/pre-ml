import preml
import re
import numpy as np


def import_data(data_name):

    with open("datasets/"+data_name+"_data.csv","r", encoding="utf-8") as f_data_in: #waveform_data
            lines = f_data_in.readlines()
            # print(type(lines))
            dataset = list()
            for line in lines:
                line = re.sub("\s+", ",", line.strip())
                parts = line.split(",")
                dataset.append(parts)

            dataset = np.array(dataset, dtype=np.float64)
#             print("dataset :")
#             dataset = dataset.T
#             print(dataset.shape)
#             print(dataset)

    with open("datasets/"+data_name+"_targets.csv","r", encoding="utf-8") as f_target_in:
            lines = f_target_in.readlines()
            targets = list()
            for line in lines:
                targets.append(line)

            targets = np.array(targets, dtype=np.int64)
#             print("targets :")
#             targets = targets.reshape(len(targets), 1) # to make it vertical if it is needed
#             print(targets.shape)
#             print(targets)
    return (dataset, targets)







def run_sample():
    try:
        (x, y) = import_data("waveform")

        solution = preml.baco(x, y, t_percent=40, heu_meth="method_1", ml_alg="knn1", iter_num=10)

        preml.draw_baco(solution)

        return True

    except Exception as e:
        raise e
        return False




def test_answer():
    assert run_sample() == True


# if __name__ == "__main__":
    # execute only if run as a script
    # run_sample()
