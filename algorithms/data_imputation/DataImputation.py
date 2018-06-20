import numpy as np
import pandas as pd
from fancyimpute import KNN


class MeanImputation:
    description = 'Mean_Imputation'

    @staticmethod
    def impute_data(data):
        df = pd.DataFrame(data)

        df = df.fillna(np.nanmean(data))

        return df[0].values.tolist()


class InterpolationImputation:
    description = 'Interpolation_Imputation'

    @staticmethod
    def impute_data(data):
        s = pd.Series(data)

        return s.interpolate().tolist()


class KNNImputation:
    description = 'KNN_Imputation'

    @staticmethod
    def impute_data(data, original_data):
        matrix = [data]

        for i in range(0, 250):
            matrix.append(original_data)

        full_data = KNN().complete(matrix)

        print(full_data)
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
