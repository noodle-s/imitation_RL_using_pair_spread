import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from per import Memory
import copy
from tqdm import tqdm

tf.compat.v1.disable_eager_execution()
Concatenate = tf.keras.layers.Concatenate(axis=-1)
mse = MeanSquaredError()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GetLoss(Loss):
    def __init__(self, loss_weight, expert_action=None, batch_size=1, out_size=1, pretrain=False):
        super().__init__()
        self.loss_weight    = loss_weight
        self.expert_action  = expert_action
        self.minibatch_size = batch_size
        self.out_size       = int(out_size)
        self.pretrain       = pretrain

    def call(self, y_true, y_pred):
        ## 1. Set MSE
        loss = tf.reduce_mean((y_true - y_pred)**2)
        if self.pretrain:
            ## 2. Compute margin_classification_loss
            margin_loss = 0
            for i in range(self.minibatch_size):
                pred_q_value   = y_pred[i]
                expert_action  = self.expert_action[i][0]
                expert_q_value = pred_q_value[expert_action]
                max_value = 0
                for action in range(self.out_size):
                    margin    = 0.0 if action == expert_action else 0.8
                    max_value = tf.maximum(pred_q_value[action] + margin, max_value)
                margin_loss += max_value - expert_q_value
            margin_loss /= self.minibatch_size
            loss += self.loss_weight * margin_loss
        return loss


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def swish(x):
    return x * tf.nn.sigmoid(x)


class DQfDNetwork:
    def __init__(self, input_len, input_feature, out_size, minibatch_size, number_of_filter):
        super(DQfDNetwork, self).__init__()
        self.input_len = input_len
        self.input_feature = input_feature
        self.out_size = out_size
        self.l2 = (10 ** -4)
        self.loss_weight = 0.23  # 0.08, 0.09
        self.minibatch_size = minibatch_size
        self.number_of_filter = number_of_filter
        self.Cross = tfrs.layers.dcn.Cross(
            kernel_initializer='VarianceScaling',
            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            bias_regularizer=tf.keras.regularizers.l2(l2=self.l2)
        )
        self.model = self._create_model()

    def _create_model(self):
        # Neural Network
        one_step_inputs = Input(shape=(self.input_feature,))
        one_cross_1 = self.Cross(one_step_inputs, one_step_inputs)
        one_cross_1 = BatchNormalization()(one_cross_1)
        one_dense_1 = Dense(one_step_inputs.shape[-1],
                            activation=mish,
                            kernel_initializer='VarianceScaling',
                            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                            bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2))(one_cross_1)

        one_dense_1 = Reshape((one_dense_1.shape[-1], 1))(one_dense_1)

        n_step_inputs = Input(shape=(self.input_len, self.input_feature,))
        n_cnn_inputs = Reshape((self.input_len, self.input_feature, 1))(n_step_inputs)
        n_cnn_inputs = BatchNormalization()(n_cnn_inputs)
        n_cnn_1 = Conv2D(self.number_of_filter, (self.input_len, 1),
                         kernel_initializer='VarianceScaling',
                         activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         activation=mish)(n_cnn_inputs)

        n_flatten_1 = Flatten()(n_cnn_1)

        n_dense_1 = Dense(n_step_inputs.shape[-1],
                          activation=mish,
                          kernel_initializer='VarianceScaling',
                          kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          activity_regularizer=tf.keras.regularizers.l2(l2=self.l2))(n_flatten_1)

        n_dense_1 = Reshape((n_dense_1.shape[-1], 1))(n_dense_1)

        concat_output = Concatenate([n_dense_1, one_dense_1])
        concat_output = Reshape((concat_output.shape[-1], concat_output.shape[-2], 1))(concat_output)

        action_1 = Conv2D(1, (2, 1),
                          kernel_initializer='VarianceScaling',
                          activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          activation=mish)(concat_output)

        action_1 = Flatten()(action_1)

        action_2 = Dense(self.out_size,
                         activation=mish,
                         kernel_initializer='VarianceScaling',
                         activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2))(action_1)

        model = Model([one_step_inputs, n_step_inputs], action_2)
        model.compile(optimizer='adam',
                      loss=GetLoss(loss_weight=self.loss_weight,
                                   batch_size=self.minibatch_size,
                                   out_size=self.out_size))

        return model


class DQfDNetwork_pretrain:
    def __init__(self, input_len, input_feature, out_size, minibatch_size, number_of_filter):
        super(DQfDNetwork_pretrain, self).__init__()
        self.input_len = input_len
        self.input_feature = input_feature
        self.out_size = out_size
        self.l2 = (10 ** -4)
        self.loss_weight = 0.23
        self.minibatch_size = minibatch_size
        self.number_of_filter = number_of_filter
        self.Cross = tfrs.layers.dcn.Cross(
            kernel_initializer='VarianceScaling',
            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            bias_regularizer=tf.keras.regularizers.l2(l2=self.l2)
        )
        self.model = self._create_model()

    def _create_model(self):
        # Neural Network
        one_step_inputs = Input(shape=(self.input_feature,))
        one_cross_1 = self.Cross(one_step_inputs, one_step_inputs)
        one_cross_1 = BatchNormalization()(one_cross_1)
        one_dense_1 = Dense(one_step_inputs.shape[-1],
                            activation=mish,
                            kernel_initializer='VarianceScaling',
                            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                            bias_regularizer=tf.keras.regularizers.l2( l2=self.l2),
                            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2))(one_cross_1)

        one_dense_1 = Reshape((one_dense_1.shape[-1], 1))(one_dense_1)

        n_step_inputs = Input(shape=(self.input_len, self.input_feature,))
        n_cnn_inputs = Reshape((self.input_len, self.input_feature, 1))(n_step_inputs)
        n_cnn_inputs = BatchNormalization()(n_cnn_inputs)
        n_cnn_1 = Conv2D(self.number_of_filter, (self.input_len, 1),
                         kernel_initializer='VarianceScaling',
                         activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         activation=mish)(n_cnn_inputs)

        n_flatten_1 = Flatten()(n_cnn_1)

        n_dense_1 = Dense(n_step_inputs.shape[-1],
                          activation=mish,
                          kernel_initializer='VarianceScaling',
                          kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          activity_regularizer=tf.keras.regularizers.l2(l2=self.l2))(n_flatten_1)

        n_dense_1 = Reshape((n_dense_1.shape[-1], 1))(n_dense_1)

        concat_output = Concatenate([n_dense_1, one_dense_1])
        concat_output = Reshape((concat_output.shape[-1], concat_output.shape[-2], 1))(concat_output)

        action_1 = Conv2D(1, (2, 1),
                          kernel_initializer='VarianceScaling',
                          activity_regularizer=tf.keras.regularizers.l2( l2=self.l2),
                          kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                          activation=mish)(concat_output)

        action_1 = Flatten()(action_1)

        action_2 = Dense(self.out_size,
                         activation=mish,
                         kernel_initializer='VarianceScaling',
                         activity_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                         bias_regularizer=tf.keras.regularizers.l2( l2=self.l2),
                         kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2))(action_1)

        expert_action = Input(shape=(1,), dtype=tf.int32)
        model = Model([one_step_inputs, n_step_inputs, expert_action], action_2)
        model.compile(optimizer='adam',
                      loss=GetLoss(self.loss_weight, expert_action, self.minibatch_size, self.out_size, pretrain=True))
        return model


class DQfDAgent:
    def __init__(self, train_env, test_env, demo, params, time):
        self.n_EPISODES = params['n_episode']
        self.train_env = train_env
        self.test_env = test_env
        self.discount_factor = params['discount_factor']
        self.target = params['target']
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.03
        self.input_feature = train_env.observation_space.shape[1]
        self.action_size = train_env.action_space.high[0] + 1
        self.n_step = params['n_step']
        self.frequency = params['frequency']  # target network update frequency
        self.demo = demo
        self.minibatch_size = params['minibatch_size']
        self.number_of_filter = params['number_of_filter']
        self.pretrain_network = DQfDNetwork_pretrain(self.n_step, self.input_feature, self.action_size,
                                                     self.minibatch_size, self.number_of_filter).model
        self.policy_network = DQfDNetwork(self.n_step, self.input_feature, self.action_size, self.minibatch_size,
                                          self.number_of_filter).model
        self.target_network = DQfDNetwork(self.n_step, self.input_feature, self.action_size, self.minibatch_size,
                                          self.number_of_filter).model
        self.demo_buffer_size = len(self.demo['state'])
        self.replay_memory_size = self.demo_buffer_size * 2
        self.demo_memory = Memory(capacity=self.demo_buffer_size, permanent_data=len(self.demo['state']))
        self.replay_memory = Memory(capacity=self.replay_memory_size, permanent_data=len(self.demo['state']))
        self.pretrain_step = params['pretrain_step']
        self._add_demo_to_memory()  # add demo data to both demo_memory & replay_memory
        self.time = time
        self.dir_path = params['dir_path']

    def _get_action(self, state, n_step_state):
        # epsilon-greedy
        if random.random() <= self.epsilon:  # exploration
            action = random.randint(0, self.action_size - 1)

        else:  # exploitation
            state     = state.reshape(1, -1)
            predicted = self.policy_network.predict([state, n_step_state])
            action    = np.argmax(predicted, axis=1)[0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        else:
            self.epsilon = self.epsilon_min

        return action

    def _add_demo_to_memory(self):
        # add demo data to both demo_memory & replay_memory
        keys = self.demo.keys()
        for idx in range(len(self.demo['state'])):
            data = {}
            for key in keys:
                if key == 'state':
                    data[key] = (self.demo[key].iloc[idx, :])
                else:
                    data[key] = (self.demo[key][idx])

            data['state'] = np.array(data['state'])
            data['next_state'] = np.array(data['next_state'])
            data['n_step_state'] = np.expand_dims(data['n_step_next_state'], axis=0)
            data['n_step_next_state'] = np.expand_dims(data['n_step_next_state'], axis=0)

            self.demo_memory.store(data)
            self.replay_memory.store(data)

    def _get_target(self, reward, next_state, n_step_next_state, pretrain):
        # next state predict
        if pretrain:
            expert_action = np.array(next_state[:, -1]).reshape(-1, 1)
            q_next_predict_value = self.pretrain_network.predict([next_state,
                                                                  n_step_next_state,
                                                                  expert_action])

        else:
            q_next_predict_value = self.policy_network.predict([next_state,
                                                                n_step_next_state])

        greedy_choice = tf.argmax(q_next_predict_value, 1)
        predict_onehot = tf.one_hot(greedy_choice, self.action_size, 1.0, 0.0)
        q_target = self.target_network.predict([next_state,
                                                n_step_next_state])
        best_v = tf.reduce_sum(q_target * predict_onehot, 1)
        target = reward + self.discount_factor * tf.stop_gradient(best_v)

        return target

    def _batch_sampling(self, memory):
        tree_idxes, minibatch, is_weights = memory.sample(self.minibatch_size)
        minibatch = pd.DataFrame(minibatch, columns=['data']).set_index(tree_idxes)
        minibatch = minibatch.sample(frac=1)

        return tree_idxes, minibatch, is_weights

    def _get_batch_generation(self, minibatch):
        minibatch = minibatch['data']
        state = np.array(minibatch.apply(lambda x: x['state']).tolist())
        reward = np.array(minibatch.apply(lambda x: x['reward']).tolist())
        next_state = np.array(minibatch.apply(lambda x: x['next_state']).tolist())
        n_step_state = np.array(minibatch.apply(lambda x: x['n_step_state']).tolist())
        n_step_state = n_step_state.reshape((self.minibatch_size, n_step_state.shape[2], -1))
        n_step_next_state = np.array(minibatch.apply(lambda x: x['n_step_next_state']).tolist())
        n_step_next_state = n_step_state.reshape((self.minibatch_size, n_step_next_state.shape[2], -1))

        return state, reward, next_state, n_step_state, n_step_next_state

    def train_network(self, pre_train):
        actual_memory = self.demo_memory if pre_train else self.replay_memory
        tree_idxes, minibatch, is_weights = self._batch_sampling(actual_memory)
        state, reward, next_state, n_step_state, n_step_next_state = self._get_batch_generation(minibatch)
        target = self._get_target(reward, next_state, n_step_next_state, pre_train)

        history = History()
        if pre_train:
            expert_action = tf.reshape(tf.cast(state[:, -1], dtype=tf.int32), [-1, 1])
            history = self.pretrain_network.fit((state,
                                                 n_step_state,
                                                 expert_action), target,
                                                verbose=0,
                                                batch_size=self.minibatch_size,
                                                steps_per_epoch=1)

        else:
            history = self.policy_network.fit((state,
                                               n_step_state), target,
                                              verbose=0,
                                              batch_size=self.minibatch_size,
                                              steps_per_epoch=1,)

        loss_series = is_weights.reshape(-1) * np.array(history.history['loss'])
        loss_series = pd.Series(loss_series, index=tree_idxes)
        self.replay_memory.batch_update(loss_series, pre_train)  # update priorities for data in memory
        return np.mean(loss_series)

    def pretrain(self):
        print(f'---------------------- Pre-training ----------------------')
        loss_list = []
        for i in tqdm(range(self.pretrain_step)):
            loss = self.train_network(pre_train=True)
            if i % self.frequency == 0:
                self.target_network.set_weights(self.pretrain_network.get_weights())
            loss_list.append(loss)
        self.policy_network.set_weights(self.pretrain_network.get_weights())
        print(f"---------------------- pretrain end ----------------------")

        plt.plot([i for i in range(self.pretrain_step)], loss_list)
        plt.savefig(self.dir_path + '/' + f"train pretrain loss, target = {self.target}, "
                                          f"time = {self.time}.png")
        plt.show()
        plt.close()

    def _get_n_step_reward(self, reward):
        n_step_reward = 0
        reward = copy.deepcopy(reward)

        for i in range(len(reward)):
            if reward[i] == 0:
                continue
            tmp = reward[i] * (self.discount_factor ** i)
            n_step_reward += tmp

        return n_step_reward

    def train(self, pre_train):
        if not pre_train and not self.replay_memory.full():  # sampling should be executed AFTER replay_memory filled
            return

        # assert self.replay_memory.full()
        res = []

        if pre_train:
            self.pretrain()

        for episode in range(self.n_EPISODES):
            terminate           = False
            state, n_step_state = self.train_env.reset()
            episode_reward_list = []

            while not terminate:
                # memory push
                action = self._get_action(state, n_step_state)
                next_state, n_step_next_state, reward, terminate = self.train_env.step(action)

                if terminate:
                    res.append(np.mean(episode_reward_list))
                    self.train_env.reset()
                    break

                else:
                    data = {'state': state,
                            'action': action,
                            'reward': reward,
                            'next_state': next_state,
                            'terminate': terminate,
                            'n_step_state': n_step_state,
                            'n_step_next_state': n_step_next_state}

                    self.replay_memory.store(data)
                    episode_reward_list.append(reward)
                    state = next_state
                    n_step_state = n_step_next_state
                    self.train_env.render()


            ##################################################################
            ### 2. GPU is not working
            self.train_network(pre_train=False)
            ##################################################################

            if episode % self.frequency == 0:
                self.target_network.set_weights(self.policy_network.get_weights())

            print(f"{episode + 1} reward average is {np.round(np.mean(episode_reward_list), 6)}")

            self.target_network.set_weights(self.policy_network.get_weights())

        self._draw_train_reward(res)

        print(f"END train function")
        print(f"all mean reward : {np.round(np.mean(res), 6)}")

    def test(self):
        episode_reward_list = []
        terminate = False
        state, n_step_state = self.test_env.reset()

        while not terminate:
            state = np.array(state).reshape(1, -1)
            predicted = self.target_network.predict([state, n_step_state])
            action = tf.keras.backend.get_value(tf.argmax(predicted, 1)[0])
            print(predicted, action)

            next_state, next_n_step_state, reward, terminate = self.test_env.step(action)
            episode_reward_list.append(reward)

            if terminate:
                print(
                    f"test reward is {np.sum(episode_reward_list)}, "
                    f"average is {np.mean(episode_reward_list)}")
                break

            state = next_state
            n_step_state = next_n_step_state
            self.test_env.render()

        print(f"END test function")
        print(f"all mean reward : {np.mean(episode_reward_list)}")

    def _draw_train_reward(self, reward_list):
        plt.plot([i for i in range(self.n_EPISODES)], reward_list)
        plt.title(f"train reward, target = {self.target}, time = {self.time}")
        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.savefig(self.dir_path + '/' + f"train reward, target = {self.target}, time = {self.time}.png")
        plt.show()
        plt.close()
