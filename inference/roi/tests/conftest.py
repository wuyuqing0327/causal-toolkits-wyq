import numpy as np
import pandas as pd
import pytest

from causalml.dataset import synthetic_data
from causalml.dataset import make_uplift_classification
from causalml.dataset import make_uplift_regression

from .const import RANDOM_SEED, N_SAMPLE, TREATMENT_NAMES, CONVERSION, COST


@pytest.fixture(scope='module')
def generate_regression_data():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_regression(n_samples=N_SAMPLE,
                                          treatment_name=TREATMENT_NAMES,
                                          y_name=CONVERSION,
                                          cost_name=COST,
                                          random_seed=RANDOM_SEED)

        return data

    yield _generate_data


@pytest.fixture(scope='module')
def generate_classification_data():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(n_samples=N_SAMPLE,
                                              treatment_name=TREATMENT_NAMES,
                                              y_name=CONVERSION,
                                              random_seed=RANDOM_SEED)

        return data

    yield _generate_data
