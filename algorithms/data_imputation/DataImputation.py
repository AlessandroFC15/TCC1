import numpy as np
import pandas as pd
from fancyimpute import KNN


class MeanImputation:
    description = 'Mean_Imputation'
    descricao = "Média"

    @staticmethod
    def impute_data(data):
        df = pd.DataFrame(data)

        df = df.fillna(np.nanmean(data))

        return df[0].values.tolist()


class InterpolationImputation:
    description = 'Interpolation_Imputation'
    descricao = 'Interpolação'

    @staticmethod
    def impute_data(data):
        s = pd.Series(data)

        # interpolation_result = s.interpolate().dropna().tolist()
        interpolation_result = s.interpolate(limit_direction='both').tolist()

        return interpolation_result


class KNNImputation:
    description = 'KNN_Imputation_N_3'
    descricao = 'KNN'

    @staticmethod
    def impute_data(data):
        print('# imputing data KNN')
        full_data = KNN(k=3).complete(data)

        return full_data


class KNNImputation_7:
    description = 'KNN_Imputation_N_7'

    @staticmethod
    def impute_data(data):
        print('# imputing data KNN')
        full_data = KNN(k=7).complete(data)

        return full_data


# class MICEImputation:
#     description = 'MICE_Imputation'
#
#     @staticmethod
#     def impute_data(data):
#         full_data = IterativeSVD(gradual_rank_increase=False).complete(np.matrix([data]))
#
#         print(full_data)
#         return full_data
