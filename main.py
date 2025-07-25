
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

NUM_INSTRUMENTS = 50
COMM_RATE = 0.0005
MAX_POSITION = 10000
currentPos = np.zeros(NUM_INSTRUMENTS)


def getMyPosition(prcSoFar):
    global currentPos
    # numDays is the number of cols
    (_,numDays) = prcSoFar.shape
       
    if (numDays < 100):
        # not enough data yet
        # print("not enough data")
        return np.zeros(NUM_INSTRUMENTS)

    for i in range(NUM_INSTRUMENTS):
        newDf = pd.DataFrame({
            'y' : prcSoFar[i]
        })

        regressorList = []

        windows = [3, 5, 10, 15, 20, 50]
        for window in windows:
            newPriceMinusAveRegressor(window, newDf, True, regressorList)
            newPriceMinusAveRegressor(window, newDf, False, regressorList)
            newStdDevRegressor(window, newDf, regressorList)

        newDf['pctChange'] = newDf['y'].pct_change()
        # get lagged returns
        for j in range(1, 4):
            newDf[f'pctChangeLag{j}'] = newDf['pctChange'].shift(j)
            regressorList.append(f'pctChangeLag{j}')

        # get RSI or MACD??
        macdRegressor(newDf, regressorList)

        # predict on last row - save values for later
        X_last = newDf[regressorList].iloc[[-1]]

        for col in regressorList:
            newDf[col] = newDf[col].shift(1)

        newDf = newDf.dropna()

        X = newDf[[col for col in regressorList]]

        Y = newDf['pctChange']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ridge = RidgeCV(cv=5).fit(X_scaled, Y)

        # print(ridge.coef_)
        # print(ridge.alpha_)

        
        X_last_scaled = scaler.transform(X_last)
        predicted_return = ridge.predict(X_last_scaled)

        if abs(predicted_return) < COMM_RATE * 2.0:
            currentPos[i] = 0
            # print(predicted_return)
        else:
            targetReturn = 1000
            netReturn = predicted_return - COMM_RATE * 2.0
            mostRecentPrice = prcSoFar[i, -1]
            maxUnits = math.floor(MAX_POSITION / mostRecentPrice)
            currentPos[i] = maxUnits * netReturn * targetReturn
    
    return currentPos

# return growth from prePrice to newPrice
def calculatePercentChange(prePrice, newPrice):
    return (newPrice - prePrice) / prePrice

# void function
'''
    weighted - boolean true if weighted ave calculated
    windowLen - length of rolling ave
'''
def newPriceMinusAveRegressor(windowLen, df, weighted, regressorList):
    if not weighted:
        df[f'roll{windowLen}Ave'] = df['y'].rolling(window=windowLen).mean()
        df[f'priceMinus{windowLen}Ave'] = df['y'] - df[f'roll{windowLen}Ave']
        regressorList.append(f'priceMinus{windowLen}Ave')
    else:
        df[f'roll{windowLen}WeightAve'] = df['y'].ewm(span=windowLen, adjust=False).mean()
        df[f'priceMinus{windowLen}WeightAve'] = df['y'] - df[f'roll{windowLen}WeightAve']
        regressorList.append(f'priceMinus{windowLen}WeightAve')

def macdRegressor(df, regressorList):
    df['ema12'] = df['y'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['y'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    regressorList.append('macd signal')

def newStdDevRegressor(windowLen, df, regressorList):
    df[f'roll{windowLen}StdDev'] = df['y'].rolling(window=windowLen).std()
    regressorList.append(f'roll{windowLen}StdDev')
