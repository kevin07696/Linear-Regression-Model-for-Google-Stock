### Machine_Learning_Basics
#### This is a simple linear regression model that trains and predicts stock data from Google
    #quandl.ApiConfig.api_key = 'Insert here'
    df = quandl.get("WIKI/GOOGL")
#### Use quandl.get to get the dataset "WIKI/GOOGL" from quandl.com 

    print(df.head())
    print(df.tail())
#### Print the first 5 rows of the dataset assigned to df

    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#### Assign the new df with the specific columns in the brackets of df. 
#### Why are there 2 brackets?
##### The first squared-bracket is for the dataframe and the second is for the array inside

    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    forecast_col = 'Adj. Close'
#### I think 'Adj.Close' is the column we are trying to forecast

    df.fillna(-99999, inplace=True)
#### By filling nan values as -99999, we can hopefully have the classifier treat them as outliers

    forecast_out = int(math.ceil(0.01*len(df)))
    #print(forecast_out)
    ''' 1% of the dataframe; math.ceil rounds to nearest whole number; 
    int makes it a whole number; forecast_out in number of rows'''

    df['label'] = df[forecast_col].shift(-forecast_out)
#### Shift up (negative) by -forecast_out, which is 35 rows
    '''
    df['Z'] = df[Y].shift(-1)
    X	Y	Z
    0	1	2
    1	2	3
    2	3	4
    3	4	NaN
    '''
#### Using 30 day window length to predict values

    X = np.array(df.drop(['label'],1))
#### 1 represents the axis, you can also write df.drop(column = ['label'])

    X = preprocessing.scale(X)
    '''scaling only works for limited training data 
    rather than incoming new data because scaling would constantly 
    be scaled slowing things down in high frequency trading'''

    X_lately = X[-forecast_out:]
#### Remember that (-) is not the same as R where it means remove
    ''' (-) means counting from the right starting from index -1 
    X = np.arange(1,6,1)
    [1 2 3 4 5]
    print(X[-3:])
    [3 4 5]
    '''
#### X_lately is the test data

    X = X[:-forecast_out]
    '''
    print(X[:-3])
    [1 2]
    '''
#### X is the training data 

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
#### Jobs are how many times the regression trains. 
#### At -1 trains as many times as your machine can run
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    #print(accuracy)
#### Using linear Regression to find the best fit line

#### svm with kernels
    for k in ['linear','poly','rbf','sigmoid']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print(k,confidence)

    forecast_set=clf.predict(X_lately)
#### Predict the indexed rows?

    print(forecast_set, "\n", accuracy, "\n", forecast_out)
#### forecast_set is the predicted values
#### Accuracy is how close is it to X_lately or the actual values
#### forecast_out is the window length

    last_date = df.iloc[-1].name
#### Locate the last date by finding the -1 position in the dataframe
    last_unix = last_date.timestamp()
#### timestamp converts everything into seconds, so it's the last_unix in seconds
    one_day = 86400
#### Converting the day to seconds
    next_unix = last_unix + one_day
#### next_unix equals the last_unix + 86400 seconds
#### These are the days predicted after the supervised test data

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
#### Assigning the next_date with datetime fromtimestamp in seconds for next_unix
    next_unix += one_day
#### Assigning the next_unix by adding one_day which is 86400 seconds
#### Then the next_unix is converted to next_date
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#### Adding columns-1 for columns in the nan rows
#### Adding another column for index i
