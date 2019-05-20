from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# prepare sequence
x1 = [1.1,2.0,3.5,4.8]
x2 = [4.1,6.1,4.9,7.2]
y = [2.0,3.4,4.2,5.1]

X = np.transpose([x1, x2])
print("shape = ",X.shape)

Y = np.transpose([y])

model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(3, activation='linear'))
model.add(Dense(1, activation='linear'))
# Compile model
model.compile(loss='mse', optimizer='sgd')
# Fit the model
model.fit(X, Y, epochs=2000)
# evaluate the model
scores = model.evaluate(X, Y)
print(scores)

x_1 = [4.8]
x_2 = [7.2]
X_test = np.transpose([x_1, x_2])
#print("shape = ", x_test.shape)
y_predict = model.predict(X_test)
print("prediction = ", y_predict)
