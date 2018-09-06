from algoritmos_felipe.DamageDetection import K_Means, Fuzzy_C_Means, DBSCAN_Center, Affinity_Propagation, GMM, G_Means, DamageDetection
from algorithms.data_imputation.DataImputation import MeanImputation, InterpolationImputation, KNNImputation
from algorithms.constants import FEATURES_DIRECTORY, RESULTS_DIRECTORY
from algorithms.feature_extraction.feature_extraction import get_extracted_data
import json


def extract_DIs(missing_data_percentage: int, imputation_method, damage_detection_method: DamageDetection) -> list:
    all_data = get_extracted_data(missing_data_percentage, 0, imputation_method)

    learn_data = all_data[0:158, :]

    damage_detection_method.train_and_test(learn_data, all_data, 197)

    return damage_detection_method.DIs.tolist()


if __name__ == "__main__":
    algorithms = [
        {'description': 'K-Means',
         'algorithm': K_Means()},

        {'description': 'Fuzzy_C_Means',
         'algorithm': Fuzzy_C_Means()},

        {'description': 'DBSCAN_Center',
         'algorithm': DBSCAN_Center(0.09, 3)},

        {'description': 'Affinity_Propagation',
         'algorithm': Affinity_Propagation()},

        {'description': 'GMM',
         'algorithm': GMM()},

        {'description': 'G_Means',
         'algorithm': G_Means()},
    ]

    list_missing_data_percentage = [5, 7, 10, 15, 20, 25, 35, 50, 60, 70, 80, 90]
    imputation_strategies = [MeanImputation, InterpolationImputation, KNNImputation]

    dis_data = {}

    for missing_data_percentage in list_missing_data_percentage:
        dis_data[missing_data_percentage] = {}

        for imputation_strategy in imputation_strategies:
            dis_data[missing_data_percentage][imputation_strategy.description] = {}

            for damage_detection_algorithm in algorithms:
                print(">> {}% | {} | {}".format(missing_data_percentage, imputation_strategy.description, damage_detection_algorithm['description']))

                dis = extract_DIs(missing_data_percentage, imputation_strategy, damage_detection_algorithm['algorithm'])

                dis_data[missing_data_percentage][imputation_strategy.description][damage_detection_algorithm['description']] = dis

    results_filename = RESULTS_DIRECTORY

    with open(results_filename + '/DIs_full_data.json', 'w') as fp:
        json.dump(dis_data, fp)