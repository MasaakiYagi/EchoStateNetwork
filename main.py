# !python
import numpy as np
from echo_state_network import EchoStateNetwork
from input_gen import InputGenerator
import matplotlib.pyplot as plt

T = np.pi * 16
RATIO_TRAIN = 0.6
dt = np.pi * 0.01
AMPLITUDE = 0.9
NUM_INPUT_NODES = 1
NUM_RESERVOIR_NODES = 100
NUM_OUTPUT_NODES = 1
LEAK_RATE = 0.02
NUM_TIME_STEPS = int(T/dt)

def main():
    # 学習用データ生成
    i_gen = InputGenerator(0, T, NUM_TIME_STEPS)
    data = i_gen.generate_sin(amplitude=AMPLITUDE)
    num_train = int( len(data) * RATIO_TRAIN )
    train_data = data[:num_train]
    
    # モデル生成
    model = EchoStateNetwork(inputs=train_data,
        num_input_nodes=NUM_INPUT_NODES, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=NUM_OUTPUT_NODES, 
        leak_rate=LEAK_RATE)
    
    # 学習
    model.train()
    train_result = model.get_train_result()
    
    # 推論
    num_predict = int(len(data[num_train:]))
    predict_result = model.predict(num_predict)

    # 描画
    t = np.linspace(0, T, NUM_TIME_STEPS)
    plt.plot(t, data, label="inputs")
    plt.plot(t[:num_train], train_result, label="trained")
    plt.plot(t[num_train:], predict_result, label="predicted")
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()
    
if __name__=="__main__":
    main()