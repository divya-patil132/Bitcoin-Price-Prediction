# Bitcoin-Price-Prediction
A Machine Learning Approach to predict the current Bitcoin Price.Through this project, our goal is to provide valuable insights for investors, traders, and stakeholders in the cryptocurrency space, ultimately aiding in informed decision-making and maximizing profitability in Bitcoin trading and investment strategies.

First, we import the required libraires for data manipulation and data visualization for the data set into the 
environment. 

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/0b759e6c-df3b-43c4-b3b5-08a1d7e030a0)

The initial data set is small enough to directly import into the collab environment. The data set is imported 
in the format of .CSV file.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/071d69b3-e42e-49e3-9bab-2366f2f27efa)

The data set is successfully imported into the collab environment.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/6b9e7098-1336-4e1a-b4cf-c5edf83cf8a3)

The data set has a total of 2713 columns and a total of 7 rows

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/e34551e4-cb81-4695-9294-b6ab8d5bfb01)
![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/5ba39621-4fe9-4a54-b5d8-8fa67dc912c7)
![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/b5056006-f171-4af8-b15c-3361f215c59b)
![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/630ee2bb-763a-4efd-9c2e-604f95395f26)

The above picture shows that there are no missing values or null values in the data set. All the variables in 
the data set are float64, which are all numerical variables. 
Using the library matplotlib with the functions plt.plot() detailed description of the data is given below.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/bc3f9f21-ed68-42eb-953f-7c43f2389fce)

• Open
It is the most important time to check out for the price of the bitcoin at this moment because it is 
starting for the day when the open price is known. This is because the open variable in the data set 
means it the price of the bitcoin at the starting point of the bitcoin when it is created or when it is 
mined. In other words, it is the price of the bitcoin when it is created. There are a total of 2713 values 
included in the data set, with a mean value of 11311.04 and a standard deviation of open price that
equals 16106.42.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/46e11a15-7e68-407f-8dde-60e4e5960305)


1.Long short-term Memory (LSTM)
  
Data preparation 
    
Here the Closing price is chosen as the target variable and stored in a new data frame. With the help of 
Mathlab library the developer as plotted the graph of the Closing price of the data set.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/c70502ad-d9ec-4ac2-8bf9-096bcd2bda5c)


First we create a new variable Price and put the Close price as a data frame in it. With the help of plt() plot the price variable. 
Xticks() functions helps the developer to plot the values according to the range set by the developer, this 
function controls of the plot range, tick locations, and the tick labels. plt.title() is a function where it helps 
to the set the title of the graph by the developer, plt.xlabel() and plt.ylabel() allows to set the names for the 
x-axis and y-axis respectively for the developer (Vanderplas, 2017).


Model building :-

Steps involved in the process of building a model include establishing methods for data collection, 
comprehending and paying attention to what is significant in the data in order to answer the questions that 
you are posing, and creating a statistical, mathematical, or simulation model in order to acquire 
understanding and make predictions.
Here the developer will use the LSTM model with the model layer of 50 neurons and the relu activation 
function. With the help of keras.model the developer will install the sequential model, as for the activation 
function the developer will be using adam , Dense, LSTM, LeakyReLU,Dropout are the layers imported 
from Kears library. 
The number of units which the number neurons are set as 64, the learning rate is set as 0.0001, the activation 
function as sigmoid and loss function as mse (mean-squared error), the batch size as 5 and the number of 
epochs as 250.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/8ab18a90-5e32-4ef9-9121-8f07af0cf10b)


Here the epochs is 250 because the more number of times the model is trained the less overfitting we get 
which leads us to a better accuracy result. Even if the epochs is set to higher value also then the model will 
have a chance to overfit. To get the more accuracy of for the model the developer will keep changing the 
number of epochs with many different values and take the best one for the result. The developer will now 
initialize the RNN model in a sequential method by calling the sequential() function and adding the layers 
to it, so the function executes in a sequential way. 
The model.dense() function is only true network layer in which each neuron in a dense layer receives one 
output from the layer that comes before it, and this output is then supplied to all of the neurons in the layer.
It is the layer that is considered to be the most fundamental in neural networks.
Now after the model building is done the developer will use the model.summay() function to get the 
summary of the model.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/7a587ab1-84c8-445a-bc4b-c25325460fae)


 Model Training :-
 
 ![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/9c9a2157-d14c-4e1a-b3ec-2215bddf419f)

The history variable is used for getting the training data, the hyper parameters are stated above in the code 
and the shuffle is set to false. The data which the developer is working on is sequential model. If the shuffle 
is set true then the date in the data will get messed up making the values go wrong as this is not a 
chorological data.


2.Auto Aggressive Integrated Moving Average (ARIMA):-

First import the required libraries for building the ARIMA model. 

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/387fb60b-2007-4956-824f-470c354603b8)


2.1 Data Preparation:

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/ea6e3f17-5468-4686-b527-17db0a1581fa)


2.2Data Split :

now split the data into 90/10.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/4f881ea7-db79-4caf-9465-e59d519c5d68)


Traning the Data

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/5d0d64fc-0791-44e0-aad9-df14b2f116ed)


2.3 Model Building :

Now build the model. Before building the model the developer will create an empty list 
and store the length of the test_data which will be used while building the model.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/fb3e37f7-de2e-4a5c-9c4f-378fac757045)

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/9630bb2c-992f-4567-a37a-9b874eab1387)



3.Results:

Two different models to predict the values from the data set. Open and Close are 
the two column used by the developer with the help of LSTM and ARIMA model.


3.1 Model Comparison :

This section will show the clear comparison of the column and show the results between the two different 
models and recommend the best the model. 
For the LSTM algorithm the data split is done for 80/20, whereas for the ARIMA algorithm the data split 
is done for 90/10.

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/92bfae5b-54b4-4796-b39d-23d527ba3894)

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/47718d93-2e32-470c-b06c-67b92df86585)

The above table explains the bitcoin price which the data has been processed with the LSTM algorithm 
with the number of epochs of 250 with a batch size of 5. The predicted price as shown above in the figure. 

![image](https://github.com/divya-patil132/Bitcoin-Price-Prediction/assets/127880624/b57a5d77-4796-4b26-ae82-2ef75caaf52a)

The Open column in the data set is processed through ARIMA model and the above figure shows the price 
predicted. 
Comparing the results obtained from the above models and MAPE value compared the best model is 
ARIMA for bitcoin price prediction. The reason is it is a common tool in demand forecasting, such as when 
anticipating the future demand for food supply or stocks prediction and bitcoin price prediction, and one of 
the common applications of this tool is as follows: When it comes to decisions regarding the supply chain, 
the developer will now have precise rules to follow. ARIMA models can also be used to forecast the future 
price of the bitcoin by analysing historical data. This is done using the data from the past (Bora, 2021). 
The layers that are present between the input and output layers are considered to be hidden layers. Because 
of this fundamental concept, deep learning networks are frequently referred to as "black boxes," and they 
have been the subject of criticism for their lack of transparency and the inability of anyone to validate the 
accuracy of their predictions. Believe it or not, a strategy that relies on trial and error will offer the best 
results depending on the specific circumstances, given that there is no universally accepted standard for the 
number of nodes (hidden neurons) or hidden layers that should be employed.
For improving the LSTM model in this project, the data split should stay 80/20 and increase the number of 
epochs with batch size as less than the usual which is 8.
