import sys

sys.path.append('../')

import time

import numpy as np
import pandas as pd
from cfbench.cfbench import BenchmarkCF, TOTAL_FACTUAL

from growingspheres import counterfactuals
from benchmark.utils import timeout, TimeoutError

# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    output_number=1,
    show_progress=True,
    disable_tf2=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()

# The Benchmark loop
growingshpheres_current_dataset = None
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']

    # Get train data
    train_data = benchmark_data['df_oh_train']

    # Get columns info
    columns = list(train_data.columns)[:-1]

    # Get factual row as pd.Series
    factual_row = pd.Series(benchmark_data['factual_oh'], index=columns)

    # Get factual class
    fc = benchmark_data['factual_class']

    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']


    def model_wrapper(x):
        return (np.array(model.predict(x)) >= 0.5).astype(int).reshape(-1,)


    @timeout(600)
    def generate_cf():
        try:
            # Create CF using GrowingSpheres' explainer and measure generation time
            start_generation_time = time.time()
            cf_gen = counterfactuals.CounterfactualExplanation(np.array(factual_array).reshape(1, -1), model_wrapper)
            cf_gen.fit()
            cf_generation_time = time.time() - start_generation_time

            if cf_gen.enemy is None:
                cf = factual_array
            else:
                cf = cf_gen.enemy.reshape(1, -1)[0].tolist()
            if model_wrapper(np.array([cf]))[0] == 1:
                print('Found CF')
            else:
                print('No CF found')

        except Exception as e:
            print('Error generating CF')
            print(e)
            # In case the CF generation fails, return same as factual
            cf = factual_row.to_list()
            cf_generation_time = np.NaN

        # Evaluate CF
        evaluator(
            cf_out=cf,
            algorithm_name='growingshpheres',
            cf_generation_time=cf_generation_time,
            save_results=True)

    try:
        generate_cf()
    except TimeoutError:
        print('Timeout generating CF')
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='growingshpheres',
            cf_generation_time=np.NaN,
            save_results=True)
