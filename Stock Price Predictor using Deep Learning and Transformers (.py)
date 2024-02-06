#!/usr/bin/env python
# coding: utf-8

# # <div align="center"> <u> Solving the Stock Market using Machine Learning </u> </div>

# ### Description:
# 
# As described in my Project Plan, I aim to use various different Machine Learning and Deep Learning algorithms and models to predict a stock's closing price on a day based on the stocks' closing prices from 60 days (2 months) prior to the current day (though the LSTM and Transformer models would, in fact, use all the days from the start of the dataset). If I do manage to train a very accurate model, I will be able to accomplish my goal of helping investors (and the general public) make large sums of profit by trading in the stock market. By predicting a company's future stock price, one can make a decision about whether to go long (buy low now and sell high in the future) or short (sell high now through borrowing shares of the stock and buy low in the future) on the stock. Knowing the direction in which the stock will move in the future will undoubtedly lead to high amounts of profits. But, in order to maximise your profits, knowing the exact stock price in the future helps since you can make a sound judgement about how many shares of the stock you wish to buy and how much risk you are willing to take (if the model predicts the stock price in the future to be way higher than the current price, you would be willing to take more risk and buy more shares of the stock).
# 
# Before I dive into the specific details about my project, I would like to give some background about trading stocks and the stock market. It is quite well-known that a share of a company's stock essentially represents a small ownership of the public side of a company. Your ownership percentage is determined by the number of shares you own of the stock divided by the total number of shares of the stock publicly available to trade. Since the stock prices of almost all public companies tend to fluctuate a lot in both directions, investors and the general public have been buying and selling stocks for a long time now in hopes of making a positive return. With the onset of the techonological revolution and the development of areas such as Machine Learning and Artificial Intelligence, most big banks and small trading and investment firms nowadays rely on machine learning based quantitative models and computational power to trade in the stock market. 
# 
# Automation in trading (namely algorithmic trading) has become very popular in recent years, and is known to yield better results that manual or systematic trading (even though algorithms can’t really factor in market sentiment and market shocks most of the times). Although a lot of work has already been done in terms of training Machine Learning models to predict future stock prices, less has been done with regards to comparing these trained Machine Learning models. I believe that through this project, I will be able to provide the finance industry and people/organizations interested in trading with great insight into which Machine Learning models are ideal to predict future stock prices so that they can go ahead and make large sums of profit in the stock market or basically any other securities’ markets. Furthermore, a lot of people have already looked at how trends in a stock's closing price (the price of the stock when the market closes on a trading day) can be used to predict future closing prices of the stock. Even though I will be mainly looking into this, I would also like to discuss a possible extension to my project by looking at how trends in the stock's <b>High</b> price (the highest price a stock attains on a trading day) and <b>Low</b> price (the lowest price a stock attains on a trading day) can be used to predict the stock's future closing prices. This is something that a lot of professionals have not looked into, and I believe that this is something that is definitely worth considering.
# 
# Now let's get into the specific details of my project implementation. As described in my Project Plan, I will be using the `yFinance` Python library to obtain the time series stock data for a company. Note that `yFinance` is an open-source Python library that allows us to acquire stock data from the Yahoo Finance website without any cost. I will be obtaining the stock price data of one company (Morgan Stanley) for approximately the past 6-7 years and storing the data in a Pandas DataFrame. I am working with only one company here in order to keep the project simple and not make it too complicated, especially since models trained on one company's stock price data may not work well for another company's stock price prediction. Furthermore, I will be comparing the following three machine learning models in this project: Random Forest Regressor (Ensemble Model), Long Short-Term Memory Recurrent Neural Network (Deep Learning Model), and Transformer (Self-Attention-based Deep Learning Model). I will be comparing the performance of these three models based on their accuracy in predicting a particular day's stock price as well as the time it takes to set-up and train the models. 
# 
# After training and testing each of my three models, I will need to compare their performance through my three main evaluation methods. The first method would be to check how far each model correctly predicted the direction in which the stock's price would move. Since there is no pre-defined library fucntion for this in Python, I will be defining my own function for this first part of the evaluation. I would like to see an accuracy score of above 50% for each of my models in this stage of the evaluation. In the second stage of the evaluation, I will be checking how far the models' predicted stock price values are from the true stock price values. I will be achieving this by calculating the root mean squared error (rmse) score of the trained models' predictions. In the third stage, I will be comparing the time efficiency of the initialization and training of each of the four models using the `time()` function in the `time` library in Python. Overall, I would expect the Transformer model to have the highest accuracy score/rmse score since it is the most complex model out of the 3 models, whereas I would expect the Random Forest model the have the best time efficiency (fastest initialization and training) since it is the least complex model out of the 3 models.   
# 
# All of the documentation of my efforts and results will be done in this Jupyter notebook itself. My efforts will be reflected by the code I write (for importing libraries, importing the dataset, training and testing the models, etc) in the code chunks in this notebook as well as the comments and descriptions (using markdown and '#') I include amongst my code and output to make my project more comprehensible. All my results (from testing and evaluating the models) will be included and shown within the notebook itself, and I will also be following the result outputs with my own descriptive analysis of the results in markdown cells. I also plan to include several graphs and diagrams in the project in order to help visualize my initial dataset as well as visualize the output of my machine learning models.  

# ### Project Status for Prototype Submission: 
# 
# Here is what I had submitted for the 'Project Status' section of my Prototype Submission. This should provide insight into how I went about completing the project and a reminder of all the things I had left to do and eventually completed:
# 
# I have already downloaded and imported all the relevant Python libraries and modules that I will be using in my Final Project. Furthermore, I have already imported the dataset that I will be using in the project and plotted the data as a time series line graph using the Matplotlib module of Python. Note that in this Prototype Assignment, I only worked with the LSTM RNN Model. After preparing the training and testing data for the model, I went ahead and set up the LSTM RNN model network architecture using the Tensorflow module. I also then trained my LSTM RNN Model using the training set and evaluated its performance (by finding its rmse score) by running it on the testing set. 
# 
# Now that I have my basic pipeline for project set up, I don't think it should be hard to extend it to the other two models. In addition to preparing the training and testing sets suited to the other two models, I need to train, test, and evaluate the other two models as well. After I have the rmse scores of each of the three models, I can go ahead and compare the three models for their accuracy. I can also implement the other two stages of evaluation I described above: accuracy score for predicting stock price movement direction as well as time efficiency. Note that, in this project, I will only build a model surrounding 'Close' prices of the stock in order to prevent it from getting too long and complicated. Of course, I am assuming here that we conduct our trades exactly at the 'Close' prices of the stock. However, I will aim to discuss how predictive the stock's 'High' and 'Low' prices are of the stock's future 'Close' prices as a possible extension to this project.  Laslty, to add visual aid, I will plot the models' stock price predictions alongside the actual stock prices towards the end of the notebook.

# In[2]:


# Use the two code lines below to install tensorflow and yfinance libraries if they aren't already installed on your server

# !pip install tensorflow
# !pip install yfinance


# In[74]:


# This cell imports all the relevant Python libraries and modules I will be using in this Final Project

import numpy as np    # For working with and manipulating large, multi-dimensional arrays and matrices
import pandas as pd    # For data manipulation and data analysis (and the DataFrame data structure)
import matplotlib.pyplot as plt    # For embedding plots, graphs, and diagrams into the notebook
import math    # For using standard and advanced mathematical constants and functions
import yfinance as yf    # This is where we will obtain our latest stock price data from
import time    # To measure the time it takes to train each of our models

from sklearn.preprocessing import MinMaxScaler    # For normalizing our stock data ranging from 0 to 1
from sklearn import metrics    # For obtaining functions for evaluating the performance of our models 
from sklearn.model_selection import train_test_split    # For splitting our dataset into training and testing sets
from sklearn.ensemble import RandomForestRegressor    # For defining and initializing a Random Forest Regressor object
from sklearn.model_selection import RandomizedSearchCV    # For finding the best set of hyperparameters for model training 

import tensorflow as tf    # To help us work with Deep Learning models such as Neural Networks
from tensorflow import keras    # To help us build our LSTM RNN Model
from tensorflow.keras import layers    # To help us build our LSTM RNN Model


# The cell above doesn't output anything, but it is essential so that we can use several complex pre-defined functions that are challenging and time-consuming to define manually.
# 
# Note that for the Transformer model, I am planning to define my own function for building the Transformer encoder blocks
# as well as define another function to add an MLP (Multi Layer Perceptron) layer to the end of the encoder blocks so that
# we get a numeric value as an output

# In[22]:


# Using the 'download' method in yFinance to acquire the Morgan Stanley stock data from 1 Jan 2016 to 30 Nov 2022

stock_data = yf.download('MS', start='2016-01-01', end='2022-11-30')


# The output above shows us that the process of downloading and storing our Morgan Stanley stock price data has been successfully completed. This dataset is what I will be working with in my Final Project.

# In[23]:


# Outputting the first five rows of the stock price dataset

stock_data.head()


# In[24]:


# Outputting the last five rows of the stock price dataset

stock_data.tail()


# The output of the above two code cells shows us the first five rows (datapoints) and the last five rows respectively of the stock price dataset that we imported. In other words. The first code cell shows us the stock price data from 4th January 2016 to 8th January 2016 while the second code cell shows us the stock price data for 22nd November, 23rd November, 25th November, 28th November, and 29th November 2022. Note that the gaps in days exist since the stock market is not open for trading on weekends and national holidays.

# In[25]:


# Plotting the stock price movement of the Morgan Stanley stock over the past 6 years and 11 months

plt.figure(figsize=(15, 8))
plt.title('Morgan Stanley Stock Price History (Jan 2016 - Nov 2022)', fontweight = 'bold', fontsize = 16)
plt.plot(stock_data['Close'])
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Stock Price ($)', fontsize = 14)


# The above diagram shows the stock price movement of the Morgan Stanley stock over the past 6 years and 11 months (from January 2016 to November 2022). The date is plotted on the x-axis and the stock price is plotted on the y-axis. Note that the diagram was constructed using the matplotlib Python library. The trend above definitely shows considerable fluctuation in the stock price when looking at subsequent trading days. However, when you zoom out, you can definitely see that there is an overall upward trend in the stock price of Morgan Stanley, and we would expect the company's stock price to rise in the future as well (albeit with considerable fluctuations). Also, note that I am only plotting the closing prices of the stock on each trading day. 

# Before we get into our model training and testing, let's first define our profit evaluation metric function that we will be using to check if each of our trained models correctly predicts the direction in which the stock price will move. This corresponds to our first evaluation metric discussed in the `Description` section of the project 

# In[99]:


def profit_eval_metric(y_test, predict):
    
    total = 0
    n = len(y_test)
    
    for i in range(1,n):
        if(y_test[i] > y_test[i-1]):
            if(predict[i] > predict[i-1]):
                total += 1
        elif(y_test[i] < y_test[i-1]):
            if(predict[i] < predict[i-1]):
                total += 1
        else:
            total += 1
    
    direction_acc = (total/(n-1)) * 100
    return direction_acc


# Now, let's first go ahead and set up, train, and test our Random Forest Regressor first, and then do the same with our LSTM and Transformer models next.

# In[149]:


# Preparing the training data (data pre-processing) for Random Forest Regressor training

stock_data_rf = stock_data

close_prices_rf = stock_data_rf['Close']
stock_values_rf = close_prices_rf.values    # Getting the column values of the 'Close' column
training_length_rf = math.ceil(len(stock_values_rf) * 0.75)    # math.ceil function rounds up the value to an integer

training_data_rf = close_prices_rf[0: training_length_rf]

X_train_rf = []
y_train_rf = []

for i in range(60, len(training_data_rf)):
    X_train_rf.append(training_data_rf[i-60:i])    # Splitting the training data into the X_train and y_train set
    y_train_rf.append(training_data_rf[i])    # Note that there are 60 values of X for every value of y


# The above cell has no output, but it is vital for us since we completely prepared our training dataset for training our Random Forest Regressor model. Note that we are only looking at the 'Close' price of the stock at this stage of the project, and I will look at how the 'High' and 'Low' prices of stocks can predict future 'Close' prices in later stages. Furthermore, note that I am not using train_test_split to split the data into training and testing sets here since I am working with time series data here, and I am using previous days' stock price data to predict a particular day's stock price.

# In[150]:


# Preparing the testing data for Random Forest Regressor model performance testing

testing_data_r = normalized_stock_data_rf[training_length_rf-60: , : ]
X_test_rf = []
y_test_rf = stock_values_rf[training_length_rf:]    # Note that we don't need a loop to create our y_test set

for i in range(60, len(testing_data)):
  X_test_rf.append(testing_data[i-60:i, 0])    # Again, there are 60 values of X for every value of y


# The above cell has no output, but it is vital for us since we completely prepared our testing dataset for testing our Random Forest Regressor model's performance

# In[155]:


# Hyperparameter tuning for the Random Forest Regressor model

grid_params_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 15, 1),  
'min_samples_split': [2, 4, 6, 8, 10], 
'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
'bootstrap': [True, False], 
'random_state': [10, 20, 30, 40]
}

model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
rscv = RandomizedSearchCV(estimator=model, scoring = 'accuracy', param_distributions=grid_params_rf, cv=3, n_jobs=-1, verbose=2, n_iter=5)
rscv_fit = rscv.fit(X_train_rf, y_train_rf)
best_parameters = rscv.best_params_
print(best_parameters)


# We ran our hyperparameter tuning for the Random Forest Regressor model above using the RandomizedSearchCV() function from scikit-learn. Note that the Random Forest Regressor has 6 main hyperparameters, and the optimal combination that we obtained is {'random_state': 30, 'n_estimators': 100, 'min_samples_split': 8, 'min_samples_leaf': 11, 'max_depth': 12, 'bootstrap': True}. 

# In[156]:


# Training the Random Forest Regressor model using the training set (based on the optimal Hyperparameter values found above)

start = time.time()

model_rf = RandomForestRegressor(n_estimators=100, random_state=30, min_samples_split=8, min_samples_leaf=11, max_depth=12, bootstrap=True)
model_rf.fit(X_train_rf, y_train_rf)

end = time.time()


# In[159]:


# Testing and evaluating the performance of the trained Random Forest Regressor model

predict_rf = model_rf.predict(X_test_rf)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predict_rf))
direction_accuracy = profit_eval_metric(y_test, predict_rf)
print("Direction Accuracy:", direction_accuracy)
print("Root Mean Squared Error:", rmse_rf)
print("Time To Train:", end-start)


# As seen above, we arrive at a direction accuracy score of about 46.1%. This means that we make a profit (or break even) during 46.1% of our trades of the Morgan Stanley stock if we use the Random Forest Regressor. This is not a very high percentage and a poor result for this model, though we will wait until we train the LSTM RNN and Transformer models before making any conclusions.
# 
# Furthermore, we obtain an RMSE score of about 65.8 for the trained Random Forest Regressor model on the stock data. This RMSE score is very high relative to our actual stock value data and it is rather surprising, which shows that the Random Forest Regressor may not the right choice for predicting time series data like our's. We might want to look into more Deep Learning model options such as the LSTM RNN and Transformer models. However, we do obtain a training time of about 5.50 seconds for the Random Forest Regressor. Even though this may seem extremely efficient at first, we still do not know how the training time compares to that of the other models we will be training. Furthermore, the low training time might be a good indicator that the Random Forest Regressor is not fitting to our data well.

# Now that we have completed training and testing our Random Forest Regressor model on the Morgan Stanley stock data, let's move on to the Long Short-Term Memory Recurrent Neural Network (LSTM RNN)

# In[92]:


# Preparing the training data (data pre-processing) for LSTM RNN model training

close_prices_lstm = stock_data['Close']
stock_values_lstm = close_prices_lstm.values    # Getting the column values of the 'Close' column
training_length_lstm = math.ceil(len(stock_values_lstm) * 0.75)    # math.ceil function rounds up the value to an integer

normalizer_lstm = MinMaxScaler(feature_range = (0,1))    # Normalizing the stock price values
normalized_stock_data_lstm = normalizer_lstm.fit_transform(stock_values_lstm.reshape(-1,1))    # Reshaping normalized data into 2-D array   
training_data_lstm = normalized_stock_data_lstm[0: training_length_lstm, :]

X_train_lstm = []
y_train_lstm = []

for i in range(60, len(training_data_lstm)):
    X_train_lstm.append(training_data_lstm[i-60:i, 0])    # Splitting the training data into the X_train and y_train set
    y_train_lstm.append(training_data_lstm[i, 0])    # Note that there are 60 values of X for every value of y

X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)    # We need numpy arrays for LSTM model training
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))    # Reshaping to 3-D array for LSTM training 


# The above cell has no output, but it is vital for us since we completely prepared our training dataset for training our LSTM RNN model. Note that we are only looking at the 'Close' price of the stock at this stage of the project. Furthermore, note that I am not using train_test_split to split the data into training and testing sets here since I am working with time series data here, and I am using previous days' stock price data to predict a particular day's stock price. Also, note that LSTM model training requires the training set to be three-dimensional

# In[93]:


# Preparing the testing data for LSTM RNN model performance testing

testing_data_lstm = normalized_stock_data_lstm[training_length_lstm-60: , : ]
X_test_lstm = []
y_test_lstm = stock_values_lstm[training_length_lstm:]    # Note that we don't need a loop to create our y_test set

for i in range(60, len(testing_data_lstm)):
  X_test_lstm.append(testing_data_lstm[i-60:i, 0])    # Again, there are 60 values of X for every value of y

X_test_lstm = np.array(X_test_lstm)    # We need numpy arrays for LSTM model testing
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))    # Reshaping to 3-D array for LSTM training


# The above cell has no output, but it is vital for us since we completely prepared our testing dataset for testing our LSTM RNN model's performance

# In[94]:


# Using Tensorflow to set up our LSTM RNN model network architecture

model_lstm = keras.Sequential()    # Since we are working with sequential time series data
model_lstm.add(layers.LSTM(100, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))    # Adding the LSTM layers
model_lstm.add(layers.LSTM(100, return_sequences=False))
model_lstm.add(layers.Dense(25))    # To incorporate the activation function in our LSTM RNN Model
model_lstm.add(layers.Dense(1))     # Contains only one network unit so that we obtain only a single output
model_lstm.summary()


# The above code cell output tells us some basic properties of the LSTM RNN model that we built. It has four layers, two of which are LSTM RNN layers while the other two are densely connected neural network layers. We can see that the total number of trainable parameters are 123,751. This high number tells us that our model is quite complex and sophisticated, and would probably yield better accuracy than the Random Forest Regressor

# In[96]:


# Training our LSTM RNN Model

start = time.time()

model_lstm.compile(optimizer='adam', loss='mean_squared_error')    # We are using the Adam Optimizer and the MSE loss function    
model_lstm.fit(X_train_lstm, y_train_lstm, batch_size = 1, epochs = 3)    

end = time.time()


# The above output shows the training phase of the LSTM RNN Model. Note that we have a batch size of 1 and we train in 3 epochs (which means we train our LSTM RNN model on the training set 3 times). As we can see, the loss of our trained model falls slightly at every training epoch, which shows that our model is fitting better to the training data at every epoch. We limit our training epochs to 3 in order to prevent our model from overfitting to the training dataset

# In[101]:


# Evaluating our LSTM RNN model's performance on the testing data

price_preds_lstm = model_lstm.predict(X_test_lstm)
price_preds_lstm = normalizer_lstm.inverse_transform(price_preds_lstm)
rmse_lstm = np.sqrt(np.mean(price_preds_lstm - y_test_lstm)**2)    # Calculating the root mean squared error of the model predictions
direction_accuracy = profit_eval_metric(y_test_lstm, price_preds_lstm)
print("Direction Accuracy:", direction_accuracy)
print("Root Mean Squared Error:", rmse_lstm)
print("Time To Train:", end-start)


# As seen in the output above, we obtain a direction accuracy score of about 52.1% for our trained LSTM RNN model. This score is higher than that of the Random Forest Regressor. We also see that we get an RMSE value of about 2.82 for the LSTM RNN model, which is way lower than that of the Random Forest Regressor model, thus telling us that the LSTM RNN model is a much more accurate predictor of future stock prices. However, note that it took a much longer time to train the LSTM RNN model (about 165.6 seconds) compared to the Random Forest Regressor.   

# Next, let's move on to the Transformer model

# In[102]:


# Preparing the training data (data pre-processing) for Transformer model training

close_prices_T = stock_data['Close']
stock_values_T = close_prices_T.values    # Getting the column values of the 'Close' column
training_length_T = math.ceil(len(stock_values_T) * 0.75)    # math.ceil function rounds up the value to an integer

normalizer_T = MinMaxScaler(feature_range = (0,1))    # Normalizing the stock price values
normalized_stock_data_T = normalizer_T.fit_transform(stock_values_T.reshape(-1,1))    # Reshaping normalized data into 2-D array   
training_data_T = normalized_stock_data_T[0: training_length_T, :]

X_train_T = []
y_train_T = []

for i in range(60, len(training_data_T)):
    X_train_T.append(training_data_T[i-60:i, 0])    # Splitting the training data into the X_train and y_train set
    y_train_T.append(training_data_T[i, 0])    # Note that there are 60 values of X for every value of y

X_train_T, y_train_T = np.array(X_train_T), np.array(y_train_T)    # We need numpy arrays for Transformer model training


# The above cell has no output, but it is vital for us since we completely prepared our training dataset for training our Transformer model. Note that we are only looking at the 'Close' price of the stock at this stage of the project. Furthermore, note that I am not using train_test_split to split the data into training and testing sets here since I am working with time series data here, and I am using previous days' stock price data to predict a particular day's stock price.

# In[103]:


# Preparing the testing data for Transformer model performance testing

testing_data_T = normalized_stock_data_T[training_length_T-60: , : ]
X_test_T = []
y_test_T = stock_values_T[training_length_T:]    # Note that we don't need a loop to create our y_test set

for i in range(60, len(testing_data_T)):
  X_test_T.append(testing_data_T[i-60:i, 0])    # Again, there are 60 values of X for every value of y

X_test_T = np.array(X_test_T)    # We need numpy arrays for Transformer model testing


# The above cell has no output, but it is vital for us since we completely prepared our testing dataset for testing our Transformer model's performance

# In[111]:


# Defining the function that sets up the Transformer encoder model architecture 

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout = 0):
    
    L = layers.LayerNormalization(epsilon=1e-6)(inputs)    # The Normalization layer is essentially an embedding layer
    
    # Adding the "Attention" layer to the architecture 
    L = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(L, L)
    L = layers.Dropout(dropout)(L)
    res = L + inputs
    
    # Adding a feed-forward part to the architecture  
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# The above function is what we will use to set up the model architecture of the Transformer Encoder. Note that we are using three different layers in our Transformer: Embedding Layer (through a Normalization layer), Attention Layer (this is what makes the Transformer model unique among the three models), and the Feed Forward Layer (to add more depth and complexity to the Transformer model)

# In[112]:


# Defining the function that actually builds the Transformer model

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    
    inputs = keras.Input(shape=input_shape)    # To convert the inputs into a form that the model understands
    x = inputs
    
    for _ in range(num_transformer_blocks):    # This is for stacking multiple transformer encoder blocks in our model
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) # To convert the outputs of the Transformer Encoder part of our model to a vector of features for each data point in the current batch
    
    for dim in mlp_units:    # This is the Multi Layer Perceptron part of the Transformer model 
        x = layers.Dense(dim, activation="elu")(x)    # The activation function 'elu' works best for our purposes in this project
        x = layers.Dropout(mlp_dropout)(x)   
        
    outputs = layers.Dense(1, activation="linear")(x)    # This is a final layer that we add to the Transformer model
    return keras.Model(inputs, outputs)


# The above model actualy goes ahead and builds the Transformer model that we will be training. It combines several Transformer encoders and also adds a Pooling layer and a Multi-Layer Perceptron layer (to add more complexity to the Transformer and to obtain a numeric value as an output to our Transformer model) to the end of the model. 

# In[113]:


def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


# The above function is a learning rate scheduler that finds the optimal learning rate to minimise the loss function of our Transformer model.

# In[117]:


# Training the Transformer model

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), keras.callbacks.LearningRateScheduler(lr_scheduler)]
input_shape = X_train_T.shape[1:]

model_T = build_model(
    input_shape,
    head_size=46,    # Setting the embedding size for the attention layer
    num_heads=60,     # Setting the number of attention heads in the attention layer
    ff_dim=55,    # Setting the hidden layer size in the feed forward network inside the Transformer model
    num_transformer_blocks=5,
    mlp_units=[256],
    mlp_dropout=0.4,
    dropout=0.14,
)

# As seen below, we are using the Mean Squared Error loss function and the Adam optimizer to train our Transformer model
model_T.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["mean_squared_error"])

start = time.time()

training_T = model_T.fit(
    X_train_T,
    y_train_T,
    validation_split = 0.2,
    epochs = 5,
    batch_size = 20,
    callbacks = callbacks,
)

end = time.time()


# The above code trains our Transformer model by minimizing the Mean Squared Error loss function using the Adam optimizer. Note that we train in 5 epochs (we train on the training set 5 times).

# In[120]:


# Testing and evaluating the performance of our trained Perceptron model on the testing data

predicted_T = model_T.predict(X_test_T)
predicted_T = normalizer_T.inverse_transform(predicted_T)
rmse_T = np.sqrt(np.mean(predicted_T - y_test_T)**2)    # Calculating the root mean squared error of the model predictions
direction_accuracy = profit_eval_metric(y_test_T, predicted_T)
print("Direction Accuracy:", direction_accuracy)
print("Root Mean Squared Error:", rmse_T)
print("Time To Train:", end-start)


# As seen above, the RMSE score of the Transformer (about 2.32) is slightly lower than that of the LSTM RNN Model (about 2.82). This implies that the Transformer is slightly more accurate than the LSTM RNN Model at predicting future stock prices (and thus is more accurate than the Random Forest Regressor). Furthermore, the direction accuracy of the Transformer model is also slightly higher than that of the LSTM RNN Model (and thus higher than that of the Random Forest Regressor), though direction accuracy is not the best indicator of the power of a price prediction model since it tends to be very random and vary across repetitions of the same training and testing process. However, the main drawback of the Transformer model is its training time (610 seconds), which is way more than that of the LSTM RNN model and the Random Forest Regressor.

# Now that we are done with training, testing, and evaluating the three models, let's visualize the results of each of the three models

# In[160]:


# Visualizing the stock price predictions of the trained Random Forest Regressor model

data = stock_data_rf.filter(['Close'])
train = data[:training_length_rf]
validation = data[training_length_rf:]
validation['Predictions'] = predict_rf
plt.figure(figsize=(16,8))
plt.title('Random Forest Regressor Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# The above result seems very weird at first, since it seems like our trained Random Forest Regressor is predicting the same stock price (which is very different from the actual stock price) every time. However, when we look at how a Random Forest Regressor works, it becomes more sophisticated as more features are added to the dataset. Since our training data contains only one feature (the 'Close' price), no wonder our Random Forest Regressor didn't work in the first place. Therefore, we can conclude that Random Forest Regressors are probably not the best choice for time series data prediction.

# In[161]:


# Visualizing the stock price predictions of the trained LSTM RNN model

data = stock_data.filter(['Close'])
train = data[:training_length_lstm]
validation = data[training_length_lstm:]
validation['Predictions'] = price_preds_lstm
plt.figure(figsize=(16,8))
plt.title('LSTM RNN Model Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[162]:


# Visualizing the stock price predictions of the trained Transformer model

data = stock_data.filter(['Close'])
train = data[:training_length_T]
validation = data[training_length_T:]
validation['Predictions'] = predicted_T
plt.figure(figsize=(16,8))
plt.title('Transformer Model Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# ### Results and Conclusion
# 
# As seen in the two figures above, both the LSTM RNN model and the Transformer model produce highly accurate results. The prediction of the stock prices in the testing set very closely match and capture the movement and fluctuating in the actual stock prices in the testing set. This is was made possible due to the improved memory capacity of an LSTM RNN model as well as the "self-attention" mechanism of a Transformer model. Note that the reasoning for why the Random Forest Regressor model produces terrible results is already provided above, so we will not cover this in this section.
# 
# Even though both figures above depict a highly accurate prediction of future stock prices (from early 2021 to late 2022), we see from the evaluation results that the trained Transformer model has a higher RMSE and direction accuracy score than the LSTM RNN Model (and thus the Random Forest Regressor Model). This implies to us that not only are we more likely to make a profit if we use a Transformer model to predict future prices of a stock, but we will be able to better prepare a trading strategy and measure our risk while conducting trades in order to maximise our profits. However, the main drawback of the Transformer model is the huge amount of time it takes to train it. Nevertheless, this should be a sacrifice one should be willing to make in order to get extremely high accuracy scores. If time is a huge factor, then the LSTM RNN model should suffice since it takes about a quarter of the time to train compared to the Transformer model, and its accuracy score is only slightly lower than the Transformer model.
# 
# Therefore, based on the results I obtained in this project, I would suggest that interested stakeholders and people can choose to use either the Transformer or the LSTM RNN model (based on their preferences and constraints) to predict future stock prices and make a profit since both are strong predictors of the same. Even though a Random Forest Regressor may not work well with just one feature or predictor, if we train it using several features (including 'Volume', 'High' price, 'Low' price, etc), it may give us promising results along with the fact that it has a very low training time. This could be a possible extension to this study. 

# ### Implications
# 
# Algorithmic trading and automation in financial markets has come a long way and is becoming more popular day by day. However, while several trading firms and banks use programming and AI/ML to make large sums of money in the financial markets through algorithmic trading, they usually forget about how can they optimize their earnings through choosing the right Machine Learning model for their trading algorithms. This particular project helps with exactly this question: for extremely high accuracy, go with the Transformer model. If you don't want to spend too much time on model training, then the LSTM RNN model should suffice. However, if you are really on a time crunch and in a hurry, training a random forest regressor with multiple features to predict future stock prices could be a good idea. These three different situations cover all the possible situations any firm or individual trader could be in, since accuracy and time are the two most important factors when it comes to making money through trading.
# 
# Therefore, by obtaining results that may be of interest to everyone, I am appealing to the entire finance industry involved with trading financial assets and helping them optimize the profits they make in a given period of time through trading. Note also that my project and conclusions are not limited to just stocks. Since prices are the main features and the response variable of my models, this project can be applied to essentially any popular financial asset (such as bonds, crypto currencies, or even financial derivatives). 

# ### Extensions and Future Work
# 
# Other than the possible extension of this study regarding training a Random Forest Regressor using multiple relevant features to predict future stock prices, another very important extension could be using factors other than just the "Close" price of a stock to predict future prices of the stock. Several other features and attributes of a stock are predictive and hold statistically significant information about the future price of the stock. Two strong candidates are the 'High' and 'Low' prices of a stock on each trading day. If the difference between the 'High' price of a stock and its closing price is much greater than the difference between 'Low' price of a stock and its closing price, then we would probably expect the stock's price to rise in the future (and vice versa). Training our three models using the 'High' and 'Low' prices of stocks every trading day in order predict future stock 'Close' prices could be an interesting extension to this project (again assuming that we excecute our trades at the end of each trading day).

# ### Related Work
# 
# [1] Biswal, A. (2022, November 15). Stock price prediction using machine learning: An easy guide: Simplilearn. Simplilearn.com. Retrieved December 18, 2022, from https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning 
# 
# [2] Adusumilli, R. (2020, January 29). Predicting stock prices using a Keras LSTM model. Medium. Retrieved December 18, 2022, from https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233 
# 
# [3] K. C, S. A. N. A, A. P, P. R, M. D and V. B. D, "Predicting Stock Prices Using Machine Learning Techniques," 2021 6th International Conference on Inventive Computation Technologies (ICICT), 2021, pp. 1-5, doi: 10.1109/ICICT50816.2021.9358537.
# 
# [4] Jigar Patel, Sahil Shah, Priyank Thakkar, K Kotecha, Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques, Expert Systems with Applications, Volume 42, Issue 1, 2015, Pages 259-268, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.07.040.
# 
# [5] Mehar Vijh, Deeksha Chandola, Vinay Anand Tikkiwal, Arun Kumar, Stock Closing Price Prediction using Machine Learning Techniques, Procedia Computer Science, Volume 167, 2020, Pages 599-606, ISSN 1877-0509, https://doi.org/10.1016/j.procs.2020.03.326.
