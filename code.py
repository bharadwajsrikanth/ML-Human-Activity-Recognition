import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import sklearn.neural_network as nn

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

test = shuffle(test)
train = shuffle(train)

trainData = train.drop('Activity', axis=1).values
trainLabel = train.Activity.values

testData = test.drop('Activity', axis=1).values
testLabel = test.Activity.values

encoder = preprocessing.LabelEncoder()

encoder.fit(trainLabel)
trainLabelE = encoder.transform(trainLabel)

encoder.fit(testLabel)
testLabelE = encoder.transform(testLabel)

mlpSGD = nn.MLPClassifier(hidden_layer_sizes=(90,),
                          max_iter=1000, alpha=1e-4,
                          solver='sgd', verbose=10,
                          tol=1e-19, random_state=1,
                          learning_rate_init=.001)

mlpADAM = nn.MLPClassifier(hidden_layer_sizes=(90,),
                           max_iter=1000, alpha=1e-4,
                           solver='adam', verbose=10,
                           tol=1e-19, random_state=1,
                           learning_rate_init=.001)

mlpLBFGS = nn.MLPClassifier(hidden_layer_sizes=(90,),
                            max_iter=1000, alpha=1e-4,
                            solver='lbfgs', verbose=10,
                            tol=1e-19, random_state=1,
                            learning_rate_init=.001)

nnModelSGD = mlpSGD.fit(trainData, trainLabelE)

nnModelADAM = mlpADAM.fit(trainData, trainLabelE)

nnModelLBFGS = mlpLBFGS.fit(trainData, trainLabelE)

train_df = train.drop("subject", axis=1)
test_df = test.drop("subject", axis=1)

def get_all_data():
    train_values = train_df.values
    test_values = test_df.values
    np.random.shuffle(train_values)
    np.random.shuffle(test_values)
    X_train = train_values[:, :-1]
    X_test = test_values[:, :-1]
    y_train = train_values[:, -1]
    y_test = test_values[:, -1]
    return X_train, X_test, y_train, y_test






