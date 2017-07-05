import pickle

import matplotlib.pyplot as plt
from net import *



def save_to_pickle(my_object, file_name):
    with open(file_name, 'wb') as output:
        pickle.dump(my_object, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    x_train, y_train = pickle.load(open('train.pkl',mode='rb'))
    nn = NeuralNetMLP(n_output=36,
                      n_features=x_train.shape[1],
                      n_hidden=63,
                      l2=0.2,
                      l1=0.0019,
                      epochs=800,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=86,
                      shuffle=True,
                      random_state=1)
    nn.fit(x_train,y_train,print_progress=True)
    save_to_pickle(nn,'model_10-06.pkl')
    print(x_train.shape)
