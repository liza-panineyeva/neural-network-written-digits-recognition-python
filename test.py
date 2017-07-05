import pickle
import predict
from datetime import datetime
import numpy as np

if __name__ == "__main__":
    x_test, y_true = pickle.load(open('train.pkl', mode='rb'))
    x_test, y_true = x_test[5000:7500], y_true[5000:7500]
    print(datetime.now())
    y_predict = predict.predict(x_test)
    print(datetime.now())
    print("rozmiar ypredict", y_predict.shape)
    print(np.count_nonzero(y_predict == y_true) / 2500)
