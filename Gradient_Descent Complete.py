
#import libraries
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import math

#Import the Boston Dataset and differentiate Features(data) and Targets
print 'Loading Boston Dataset.....'
boston = datasets.load_boston()
data=boston.data
target=boston.target
tr=data.shape[0]  #total rows in the data set i.e. 506
tc=data.shape[1]  #total columns in the data set i.e 13


#Combining the Features and Target
#We are appending to 'fulltable' in such a way that 14th Feature of the data is actually the target
fulltable=np.empty([tr,tc+1])
for i in range (0,tr):
	fulltable[i][tc]=target[i]
	for j in range(0,tc):
		fulltable[i][j]=data[i][j]
#print fulltable #to check the table-- First 13 Columns are Features, 14th Column is the Target



#NORMALIZING THE DATASET
print 'Normalizing the Dataset.....'
#Find min and max values amongst all the columns
minvals={}
maxvals={}
for j in range(0,tc+1):
	minD=fulltable[0][j]
	maxD=fulltable[0][j]
	for i in range(0,tr):
		if fulltable[i][j] < minD:
			minD=fulltable[i][j]
		if fulltable[i][j] > maxD:
			maxD=fulltable[i][j]
		minvals[j]=minD
		maxvals[j]=maxD


#Normalize all the values using Min and Max calculate in the previous step
#make a new table normaldata, which is a copy of the original dataset fulltable but with one more condition
#It is not the exact original, but normalized values of original data
normaldata = fulltable.copy()
for j in range(0,tc+1):
	for i in range(0, tr):
		 normaldata[i][j]= (normaldata[i][j] - minvals[j]) / (maxvals[j]-minvals[j])
print 'Normalizing Successful.....'


#After the data has been normalized, we split the data
#splitting data to traing and test in 90:10
print 'Splitting Dataset to Train and Test in Ratio 90:10 .....'
trainX=[]
trainY=[]
testX=[]
testY=[]
for i in range(0,tr):
	if i%10==0:
		testX.append(normaldata[i][0:tc])
		testY.append(normaldata[i][tc])
	else:
		trainX.append(normaldata[i][0:tc])
		trainY.append(normaldata[i][tc])
print 'Train Table has',len(trainX), 'rows, which is', round(float(len(trainX)*100)/(len(trainX)+len(testX)),0),'% of total values'
print 'Test Table has',len(testX), 'rows, which is', round(float(len(testX)*100)/(len(trainX)+len(testX)),0),'% of total values'


#Now we have split the original Normalized data to train and test
#But it will be easy to have the train values all in one table
#So we combine all Train features and Train targets in one table called train
#We are going to use the TABLE TRAIN to do gradient descent
train=[]
for i in range(0,len(trainX)):
    feat = trainX[i]
    tar=trainY[i]
    row=[feat,tar]
    train.append(row)

#Similarly we do it for Test Data
test=[]
for i in range(0,len(testX)):
    feat = testX[i]
    tar=testY[i]
    row=[feat,tar]
    test.append(row)

#Now that we have created the table TRAIN which contains all the Normalized Features and Target
#To access the certain feature or target, we use the following syntax-
# print train[0][0][0] #TO GET THE FIRST ROW FIRST FEATURE
# print train[0][0][1] #TO GET THE FIRST ROW SECOND FEATURE
# print train[0][1] #TO GET THE TARGET FOR FIRST ROW


#Function to predict value based on input features
def model(b0, bl,t):
    terms=[]
    for i in range(0,tc):
        s =bl[i] * t[i]
        terms.append(s)
    return b0 + sum(terms)

#function to calculate RMSE
def RMSE(trainY,predicted):
    elist=[]
    # print elist
    for k in range(0, len(predicted)):
        # print trainY[k]
        errorsq= (trainY[k]-predicted[k])**2
        elist.append(errorsq)
        # print el
    return math.sqrt(sum(elist)/float(len(predicted)))


#different learning rates
learning= [1,0.1,0.01,0.001,0.0001,0.00001]
print 'Performing gradient descent based on the learning rates', learning,'.....'


#Loops for calculating predicted values
#for each learning rate
#and 10 epochs for each learning rate
print 'Now let\'s compute RMS Errors'
for l in range(0,len(learning)):
    learning_rate=learning[l]
    b0 = 0.0
    b = [0 for x in range(tc)]
    predictions = []
    RMSEVals = []
    errorlist = []
    epoch = 0
    while epoch<10:
        predictions = []
        for i in range (0, len(train)):
            error= model(b0, b, train[i][0]) - train[i][1]
	    print error
            b0=b0- learning_rate*error
            for j in range(0, tc):
            	b[j] = b[j] - learning_rate * error * train[i][0][j]
        epoch+=1


	#We can use either Train Set or Test Set to calculate RMSE Errors
	#I am going to use the TEST Set to calculate RMSE Error
	#Calculate RMSE on the TEST SET, and check the performance of the learning curve
	for k in range(0,len(test)):
            prediction=model(b0,b,test[k][0])
            predictions.append(prediction)
        RMSEVals.append(RMSE(testY,predictions))
    if learning_rate==1:
	print 'DAMNNNNN! For LearningRate=1, RMS Error is extremely large, resulting to Inf and eventually is NAN(Not a number)'
    else:
	print'RMS Errors FOR LEARNING RATE', learning_rate, 'are',RMSEVals

   #plotting for each learning rate
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(x, RMSEVals)
    plt.title('Learning Rate' + str(learning_rate))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.show()

'''	
#TO CALUCLATE RMSE ON TRAIN DATA SET, WE CAN USE THE FOLLOWING LOOP
#AND REPLACE IT WHERE WE ARE CALUCLATING RMSE IN THE MAIN LOOP
	for i in range(0,len(train)):
            prediction=model(b0,b,train[i][0])
            predictions.append(prediction)
        RMSEVals.append(RMSE(trainY,predictions))
    print'RMS Errors FOR LEARNING RATE', learning_rate, 'are',RMSEVals
    #plotting for each learning rate
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #plt.plot(x, RMSEVals)
    #plt.title('Learning Rate' + str(learning_rate))
    #plt.xlabel('Epoch')
    #plt.ylabel('RMSE')
    #plt.show()
'''


print 'Hope you liked the graphs. Now we can analyze and choose the best learning_rate!!'


print 'WOAHHH it is 3:30 am, let\'s just call it a night!'
print 'Goodnight World!!!!'




