from StockEnv import StockTradingEnv
from DataPreprocessing import DataGet
from DQfd import DQfDAgent
from datetime import datetime
import os
import json
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')


if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def _set_params():
    params = {'period_start_list': ['2011-01-01'],  # ok
              'form_period_end_list': ['2017-12-31'],  # ok
              'demo_period_end_list': ['2012-12-31'],  # ok
              'test_period_end_list': ['2019-12-31'],  # ok
              'target': ["삼성전자", "NAVER", "LG전자", "현대차", "SK하이닉스",
                         "삼성SDI", "POSCO", "LG생활건강", "신한지주", "엔씨소프트"],  # ok
              'n_episode': 2000,  # ok
              'dist': 5,  # ok
              'mdpp_rate': 5,  # [3, 4]
              'minibatch_size': 16,  # ok
              'pretrain_step': 150,  # ok
              'transaction_cost': 0.00015,  # ok
              'tax': 0.0025,  # ok
              'n_step': 5,  # ok
              'frequency': 10,  # ok
              'discount_factor': 0.93,  # ok
              'pre_train': True,
              'nrm': 1, # ok
              'order_rate': 0.02,  # ok
              'number_of_filter': 10,
              'bal': 100000000,
              'expert_action': 'MDPP'}  # MDPP, Osciliator, RSI, MACD

    return params


def _set_path_time():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    date = dt_string.split()[0]
    time = dt_string.split()[1]
    dir_path = os.getcwd() + '//' + date

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    return dir_path, time


def _save_param(param, time):
    dir_path = param['dir_path']
    file_list = os.listdir(dir_path)
    file_list_json = list(map(lambda e: e.endswith('.json'), file_list))
    file_list = [file for file in file_list if file_list_json]

    if file_list:
        for file in file_list:
            if file.endswith('.json'):
                with open(dir_path + '//' + str(file), 'r', encoding="utf-8") as exist_file:
                    output = json.load(exist_file)

                    if not output == param:
                        with open(dir_path + '//' + time + '.json', 'w') as new_file:
                            json.dump(param, new_file, ensure_ascii=False)

            else:
                continue

    else:
        with open(dir_path + '//' + time + '.json', 'w') as new_file:
            json.dump(param, new_file, ensure_ascii=False)


def _print_info(data, demo):
    print(f"formation_period = {data.train_data['state'].index[0]} ~ {data.train_data['state'].index[-1]}")
    print(f"demonstration_period = {demo['state'].index[0]} ~ {demo['state'].index[-1]}")
    print(f"target = {data.target_stock}")
    print(f"number of pair = {len(data.train_data['state'].columns) - 5}")
    print(f"pair stock = {list(data.train_data['state'].columns).remove('rate')}")
    print(f"sum of demo reward = {sum(demo['reward'])}")
    print(f"mean of demo reward = {sum(demo['reward']) / len(demo['reward'])}")


def main():
    params = _set_params()
    params['dir_path'], time = _set_path_time()
    params['period_start'] = params['period_start_list'][0]
    params['demo_period_end'] = params['demo_period_end_list'][0]
    params['form_period_end'] = params['form_period_end_list'][0]
    params['test_period_end'] = params['test_period_end_list'][0]
    params['target'] = params['target'][0]
    # _save_param(params, time)


    data = DataGet(params=params)
    data.preprocessing_data()
    demo = data.get_demo_data()

    _print_info(data, demo)

    train_state, train_n_step_state, test_state, test_n_step_state = data.get_state()
    train_env = StockTradingEnv(train_state, train_n_step_state, data.open_target, params, time, train=True)
    test_env = StockTradingEnv(test_state, test_n_step_state, data.open_target, params, time, train=False)

    model = DQfDAgent(train_env, test_env, demo, params, time)
    model.train(params['pre_train'])
    model.test()


if __name__ == '__main__':
    main()
