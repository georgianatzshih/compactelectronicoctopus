author = "georgiana shih"
# utilised ML video by Python for Microscopists, Sreenivas Bhattiprolu: https://youtu.be/2yhLEx2FKoY

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
random.seed(1)

import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force = True)
from matplotlib import pyplot as plt

# note: currencies key
#   b = BTC = 1
#   c = CAD = 2
#   e = EUR = 3
#   g = GBP = 4
#   j = JPY = 5
#   u = USD = 6

def namingCSVs(path, filename, currency1, currency2):
    temp = os.path.join(path, filename)
    df = pd.read_csv(temp)
    match currency1:
        case "b":
            c1 = 1
        case "c":
            c1 = 2
        case "e":
            c1 = 3
        case "g":
            c1 = 4
        case "j":
            c1 = 5
        case "u":
            c1 = 6
        case _:
        # exit()
            c1 =0
    match currency2:
        case "b":
            c2 = 1
        case "c":
            c2 = 2
        case "e":
            c2 = 3
        case "g":
            c2 = 4
        case "j":
            c2 = 5
        case "u":
            c2 = 6
        case _:
            exit()

    currency_1 = list()
    currency_2 = list()

    r, c = df.shape
    print(r)
    for i in range(r):
        currency_1.append(c1)
        currency_2.append(c2)
    df['currency1'] = currency_1
    df['currency2'] = currency_2
    return (df)

# note: banks key
#   Bake = 1
#   BoA = 2
#   Barclays = 3
#   Binance = 4
#   Coinbase = 5
#   DB = 6
#   HSBC = 7
#   Huobi = 8
#   Jeffries = 9
#   Pancake = 10

def convertBanks(df, column):
    r, c = df.shape
    print(r)
    b = list()
    for i in range(r):
        match str(df[column][i]):
            case 'Bake':
                bank = 1
            case 'BankOfAmerica':
                bank = 2
            case 'Barclays':
                bank = 3
            case 'Binance':
                bank = 4
            case 'Coinbase':
                bank = 5
            case 'DeutscheBank':
                bank = 6
            case 'HSBC':
                bank = 7
            case 'Huobi':
                bank = 8
            case 'Jeffries':
                bank = 9
            case 'Pancake':
                bank = 10
            case _:
                exit()
        b.append(bank)
    banks = np.array(b)
    df[column] = banks
    return(df)


path = r"D:\[ f i l e s ]\d o c u m e n t s\u n i\y e a r   3\[ other ]\durhack\bidfx\all"

# columns: date, time (hr), open, high, low, close, volume, currency1, currency2
_1923_b_u = namingCSVs(path, "1923BTCUSD.csv", "b", "u")
_1923_e_c = namingCSVs(path, "1923EURCAD.csv", "e", "c")
_1923_e_g = namingCSVs(path, "1923EURGBP.csv", "e", "g")
_1923_e_u = namingCSVs(path, "1923EURUSD.csv", "e", "u")
_1923_g_u = namingCSVs(path, "1923GBPUSD.csv", "g", "u")
_1923_u_c = namingCSVs(path, "1923USDCAD.csv", "u", "c")
_1923_u_j = namingCSVs(path, "1923USDJPY.csv", "u", "j")

temp = [_1923_b_u, _1923_e_c, _1923_e_g, _1923_e_u, _1923_g_u, _1923_u_c, _1923_u_j]
_1923 = list()
_1923.extend(temp)

# columns: time (ms), sell price, buy price, bank
_2109_b_u = namingCSVs(path, "2109BTCUSD.csv", "b", "u")
_2109_e_c = namingCSVs(path, "2109EURCAD.csv", "b", "u")
_2109_e_g = namingCSVs(path, "2109EURGBP.csv", "b", "u")
_2109_e_u = namingCSVs(path, "2109EURUSD.csv", "b", "u")
_2109_g_u = namingCSVs(path, "2109GBPUSD.csv", "b", "u")
_2109_u_c = namingCSVs(path, "2109USDCAD.csv", "b", "u")
_2109_u_j = namingCSVs(path, "2109USDJPY.csv", "b", "u")

temp = [_2109_b_u, _2109_e_c, _2109_e_g, _2109_e_u, _2109_g_u, _2109_u_c, _2109_u_j]
_2109 = list()
_2109.extend(temp)


## assigning column titles
all_1923 = pd.DataFrame()
temp_d = list()
temp_hr = list()
temp_o = list()
temp_h = list()
temp_l = list()
temp_c = list()
temp_v = list()
temp_cu1 = list()
temp_cu2 = list()
for i in range(7):
    _1923[i].columns = ['d', 'hr', 'o', 'h', 'l', 'c','v', 'currency1', 'currency2']
    _2109[i].columns = ['ms', 's', 'b', 'B_', 'currency1', 'currency2']
    convertBanks(_2109[i], "B_")

    temp_d = temp_d + _1923[i].d.tolist()
    temp_hr = temp_hr + _1923[i].hr.tolist()
    temp_o = temp_o + _1923[i].o.tolist()
    temp_h = temp_h +_1923[i].h.tolist()
    temp_l = temp_l + _1923[i].l.tolist()
    temp_c = temp_c + _1923[i].c.tolist()
    temp_v = temp_v + _1923[i].v.tolist()
    temp_cu1 = temp_cu1 + _1923[i].currency1.tolist()
    temp_cu2 = temp_cu2 + _1923[i].currency2.tolist()





r, c = all_1923.shape
print('rows:', r, 'columns:', c)

all_2109 = pd.concat([_2109[0], _2109[1], _2109[2], _2109[3], _2109[4], _2109[5], _2109[6]])
r, c = all_1923.shape
print('rows:', r, 'columns:', c)

x = pd.DataFrame()
x['d'] = temp_d
x['hr'] = temp_hr
x['o'] = temp_o
x['h'] = temp_h
x['l'] = temp_l
x['c'] = temp_c
x['v'] = temp_v
x['currency1'] = temp_cu1
x['currency2'] = temp_cu2

r,c = x.shape
print(r,c)
y = pd.DataFrame()
y = ('d', 'hr', 'o', 'h', 'l', 'c','v', 'currency1', 'currency2')

r,c = y.shape
print(r,c)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.05, random_state = 20, stratify = y)
scaler = StandardScaler()
scaler.fit(xTrain)

xTrainScaled = scaler.transform(xTrain)
xTestScaled = scaler.transform(xTest)



# model
hln = 32
model = Sequential()
# input
model.add(Dense(128, input_dim=351, activation='relu'))
# hidden
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
model.add(Dense(hln, activation='relu'))
    # output
model.add(Dense(9, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

history = model.fit(xTrainScaled, yTrain, validation_split=0.02, epochs =200)

loss = history.history['loss']
valLoss = history.history['valLoss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, valLoss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['mae']
valAcc = history.history['valMAE']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, valAcc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions = model.predict(xTestScaled)
print("Predicted values are: ", predictions)
print("Real values are: ", yTest[:3])