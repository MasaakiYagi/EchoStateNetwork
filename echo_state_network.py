# !python
import numpy as np
from scipy import linalg

class EchoStateNetwork:
    
    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate=0.1, activator=np.tanh):
        self.inputs = inputs  # 教師データ入力
        self.log_reservoir_nodes = np.array([np.zeros(num_reservoir_nodes)])  # reservoir層のノード状態
        
        # 重み初期値
        self.weights_input = self.__generate_i2r_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self.__generate_r_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])  # 0で初期化
        
        # 各層ノード数
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes
        
        # その他設定値
        self.leak_rate = leak_rate  # 漏れ率
        self.activator = activator  # 活性化関数
    
    def train(self, lambda0=0.1):
        for input in self.inputs:
            current_state = np.array(self.log_reservoir_nodes[-1])
            self.log_reservoir_nodes = np.append(self.log_reservoir_nodes,
                [self.__get_next_reservoir_nodes(input, current_state)], axis=0)
        self.log_reservoir_nodes = self.log_reservoir_nodes[1:]  # 初期状態は削除
        self.__update_weights_output(lambda0)
    
    def get_train_result(self):
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for input in self.inputs:
            reservoir_nodes = self.__get_next_reservoir_nodes(input, reservoir_nodes)
            outputs.append(self.get_output(reservoir_nodes))
        return outputs
    
    def predict(self, length_of_sequence, lambda0=0.01):
        predicted_outputs = [self.inputs[-1]]  # 予測の最初のインプットは学習データの末尾データ
        reservoir_nodes = self.log_reservoir_nodes[-1]  # 学習の最後のリザバー層状態
        for _ in range(length_of_sequence):
            reservoir_nodes = self.__get_next_reservoir_nodes(predicted_outputs[-1], reservoir_nodes)
            predicted_outputs.append(self.get_output(reservoir_nodes))
        return predicted_outputs[1:]  # 予測起点の最初のデータ以外を返す
    
    def get_output(self, reservoir_nodes):
        return reservoir_nodes @ self.weights_output
    
    #
    # private method
    #
    
    # 入力層-リザバー層の重みを0.1 or -0.1で返す。入力層ノード数×リザバー層ノード数の行列
    def __generate_i2r_weights(self, num_i_nodes, num_r_nodes):
        return (np.random.randint(0, 2, num_i_nodes*num_r_nodes).reshape([num_i_nodes, num_r_nodes])*2-1)*0.1
    
    # リザバー層内のノード間重みをランダムに返す。リザバー層ノード数×リザバー層ノード数の行列
    def __generate_r_weights(self, num_r_nodes):
        weights = np.random.normal(0, 1, num_r_nodes*num_r_nodes).reshape([num_r_nodes, num_r_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius
    
    # リザバー層ステップ更新
    def __get_next_reservoir_nodes(self, input, current_state):
        next_state = (1 - self.leak_rate) * current_state\
                    + self.leak_rate * (np.array([input]) @ self.weights_input + current_state @ self.weights_reservoir)
        return self.activator(next_state)
    
    # 出力層の重み更新
    def __update_weights_output(self, lambda0):
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0
        inv_x = np.linalg.inv(self.log_reservoir_nodes.T @ self.log_reservoir_nodes + E_lambda0)
        self.weights_output = (inv_x @ self.log_reservoir_nodes.T) @ self.inputs