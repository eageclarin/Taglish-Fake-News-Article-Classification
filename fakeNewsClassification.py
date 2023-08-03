###IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import time
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score)

##importing the dataset
dataset = pd.read_csv('full.csv')
labels = dataset['label'] #0 - fake; 1 - legit
articles = dataset['article']
labelIndex = np.array(labels.index.values)

###PREPROCESSING
##train test split
testSize = 320 #change for testing #320 90:10 #962 70:30
randomState = 200 #change per run; 1, 10, 100, 200, 300
configure = var2 #var2, var3, var4, var5, var6, var7, var8, var9
nvTrain, nvTest, trainLabels, testLabels = train_test_split(articles, labels, test_size = testSize, 
random_state=randomState, stratify=labels) #not vectorized

##configuration
#to have 100% fake news in test data
def var2():
    print("*** full fake news")
    #not vectorized
    trainIndices = trainLabels.index[trainLabels == 0].tolist()
    testIndices = testLabels.index[testLabels == 1].tolist()

    #replace legit news in test data with fake news from train data
    for i in testIndices:
        #retrieve random index
        np.random.seed(i) #for same result
        randomIndex = np.random.choice(trainIndices)

        #swap values
        testLabels[i], trainLabels[randomIndex] = trainLabels[randomIndex], testLabels[i] #not vectorized
        nvTest[i], nvTrain[randomIndex] = nvTrain[randomIndex], nvTest[i] #not vectorized

        trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is replaced

def var3():
    print("*** full legit news")
    #not vectorized
    trainIndices = trainLabels.index[trainLabels == 1].tolist()
    testIndices = testLabels.index[testLabels == 0].tolist()

    #replace fake news in test data with legit news from train data
    for i in testIndices:
        #retrieve random index
        np.random.seed(i) #for same result
        randomIndex = np.random.choice(trainIndices)

        #swap values
        testLabels[i], trainLabels[randomIndex] = trainLabels[randomIndex], testLabels[i] #not vectorized
        nvTest[i], nvTrain[randomIndex] = nvTrain[randomIndex], nvTest[i] #not vectorized

        trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is replaced

def configTest(partition): #50:50 training data
    global testLabels
    global nvTest
    
    match partition:
        case var4: #100% fake news
            print("*** 50/50 train 100 (fake) / 0 (legit) test ")

            ##retrieve and duplicate from training data
            missingData = len(testLabels) - len(testLabels[testLabels == 0]) #compute number of missing data
            trainIndices = trainLabels.index[trainLabels == 0].tolist() #retrieve fake news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()
            
            for i in range(missingData):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)

            #retain fake news only
            testLabels = testLabels.loc[testLabels == 0]
            nvTest = nvTest.loc[testLabels.index[testLabels == 0]]
        case var5: #100% legit news
            print("*** 50/50 train 0 (fake) / 100 (legit) test ")

            ##retrieve and duplicate from training data
            missingData = len(testLabels) - len(testLabels[testLabels == 1]) #compute number of missing data
            trainIndices = trainLabels.index[trainLabels == 1].tolist() #retrieve legit news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()
            
            for i in range(missingData):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)

            #retain legit news only
            testLabels = testLabels.loc[testLabels == 1]
            nvTest = nvTest.loc[testLabels.index[testLabels == 1]]
        case var6: #70% fake news; 30% legit news
            print("*** 50/50 train 70 (fake) / 30 (legit) test")
            seventy = len(testLabels)*.70 #70% of len(test) for fake
            thirty = len(testLabels)*.30 #30% of len(test) for legit

            missingData = seventy - len(testLabels[testLabels == 0]) #70% fake; calculate missing number of fake news
            excessData = len(testLabels[testLabels == 1]) - thirty #30% legit; remove excess number of legit news

            trainIndices = trainLabels.index[trainLabels == 0].tolist() #retrieve fake news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()

            ##retrieve and copy from training data
            for i in range(int(missingData)):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert to list
            toRemoveIndices = np.where(testLabels == 1)[0] #to be removed from test data
            testLabels = testLabels.tolist()
            nvTest = nvTest.tolist()
            
            ##remove excess data
            #must be sorted from highest value to lowest
            np.random.seed(1)
            randomIndex = np.random.choice(toRemoveIndices, int(excessData), replace=False)

            for i in sorted(randomIndex, reverse=True):
                #remove from test data
                nvTest.pop(i)
                testLabels.pop(i)

                randomIndex = np.delete(randomIndex, np.where(randomIndex == i))
            
            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)
        case var7: #30% fake news; 70% legit news
            print("*** 50/50 train 30 (fake) / 70 (legit) test")
            thirty = len(testLabels)*.30 #30% of len(test) for fake
            seventy = len(testLabels)*.70 #70% of len(test) for legit

            missingData = seventy - len(testLabels[testLabels == 1]) #70% legit; calculate missing number of fake news
            excessData = len(testLabels[testLabels == 0]) - thirty #30% fake; remove excess number of legit news

            trainIndices = trainLabels.index[trainLabels == 1].tolist() #retrieve legit news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()

            ##retrieve and copy from training data
            for i in range(int(missingData)):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert to list
            toRemoveIndices = np.where(testLabels == 0)[0] #to be removed from test data
            testLabels = testLabels.tolist()
            nvTest = nvTest.tolist()
            
            ##remove excess data
            #must be sorted from highest value to lowest
            np.random.seed(1)
            randomIndex = np.random.choice(toRemoveIndices, int(excessData), replace=False)

            for i in sorted(randomIndex, reverse=True):
                #remove from test data
                nvTest.pop(i)
                testLabels.pop(i)

                randomIndex = np.delete(randomIndex, np.where(randomIndex == i))
            
            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)
        case var8: #90% fake news; 10% legit news
            print("*** 50/50 train 90 (fake) / 10 (legit) test")
            ninety = len(testLabels)*.90 #90% of len(test) for fake
            ten = len(testLabels)*.10 #10% of len(test) for legit

            missingData = ninety - len(testLabels[testLabels == 0]) #70% fake; calculate missing number of fake news
            excessData = len(testLabels[testLabels == 1]) - ten #30% legit; remove excess number of legit news

            trainIndices = trainLabels.index[trainLabels == 0].tolist() #retrieve fake news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()

            ##retrieve and copy from training data
            for i in range(int(missingData)):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert to list
            toRemoveIndices = np.where(testLabels == 1)[0] #to be removed from test data
            testLabels = testLabels.tolist()
            nvTest = nvTest.tolist()
            
            ##remove excess data
            #must be sorted from highest value to lowest
            np.random.seed(1)
            randomIndex = np.random.choice(toRemoveIndices, int(excessData), replace=False)

            for i in sorted(randomIndex, reverse=True):
                #remove from test data
                nvTest.pop(i)
                testLabels.pop(i)

                randomIndex = np.delete(randomIndex, np.where(randomIndex == i))
            
            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)
        case var9: #10% fake news; 90% legit news
            print("*** 50/50 train 10 (fake) / 90 (legit) test")
            ten = len(testLabels)*.10 #10% of len(test) for fake
            ninety = len(testLabels)*.90 #90% of len(test) for legit

            missingData = ninety - len(testLabels[testLabels == 1]) #90% legit; calculate missing number of fake news
            excessData = len(testLabels[testLabels == 0]) - ten #10% fake; remove excess number of legit news

            trainIndices = trainLabels.index[trainLabels == 1].tolist() #retrieve legit news from training data
            
            #convert to numpy array
            testLabels.to_numpy()
            nvTest.to_numpy()

            ##retrieve and copy from training data
            for i in range(int(missingData)):
                np.random.seed(i) #for same result
                randomIndex = np.random.choice(trainIndices) #retrieve random index

                #add a copy from training data
                testLabels = np.append(testLabels, trainLabels[randomIndex]) #label
                nvTest = np.append(nvTest, nvTrain[randomIndex]) #article
                trainIndices = np.delete(trainIndices, np.where(trainIndices == randomIndex)) #data is added

            #convert to list
            toRemoveIndices = np.where(testLabels == 0)[0] #to be removed from test data
            testLabels = testLabels.tolist()
            nvTest = nvTest.tolist()
            
            ##remove excess data
            #must be sorted from highest value to lowest
            np.random.seed(1)
            randomIndex = np.random.choice(toRemoveIndices, int(excessData), replace=False)

            for i in sorted(randomIndex, reverse=True):
                #remove from test data
                nvTest.pop(i)
                testLabels.pop(i)

                randomIndex = np.delete(randomIndex, np.where(randomIndex == i))
            
            #convert back to series
            testLabels = pd.Series(testLabels)
            nvTest = pd.Series(nvTest)

#var2() #remove/add comment
#var3() #remove/add comment
configTest(configure) #remove/add comment

##count vector for training
countVector = CountVectorizer()
bowTrain = countVector.fit_transform(nvTrain) #bag of words
bowTest = countVector.transform(nvTest)
bowTrain = np.array(bowTrain.todense())
bowTest = np.array(bowTest.todense())
train = bowTrain
test = bowTest

###APPLYING CLASSIFIERS (1)
## NAIVE BAYES ALGORITHM
nb_startTime = time.time()

NBmodel = MultinomialNB().fit(train, trainLabels) ##train the model
NBprediction = NBmodel.predict(test) ##making predictions

##calculate running time
nb_endTime = time.time()
nb_elapsedTime = nb_endTime - nb_startTime

###EVALUATION (1)
##evaluate naive bayes
print('NAIVE BAYES ALGORITHM')
print('Test Size:', testSize)
print('Accuracy:', accuracy_score(testLabels, NBprediction))
print('F1 score:', f1_score(testLabels, NBprediction, average="macro"))
print('Runtime:', nb_elapsedTime)

##misclassified in prediction
NBpredLabel = [item for item in NBprediction] #predicted label of testing
NBtestLabels = [item for item in testLabels] #primary label of test
NBtestFeatures = [item for item in nvTest] #primary features of test
misclassified = [index for index, elem in enumerate(NBpredLabel) if elem != NBtestLabels[index]] #after testing
l_testAgain = []
f_testAgain = []
for i in sorted(misclassified, reverse=True): #add every misclassified to testAgain
    l_testAgain.append(NBtestLabels[i]) #add label to testAgain
    f_testAgain.append(NBtestFeatures[i]) #add feature to testAgain

    NBprediction = np.delete(NBprediction, i) #delete misclassified data from test prediction 

classified = [index for index, elem in enumerate(NBpredLabel) if elem == NBtestLabels[index]]
for i in sorted(classified, reverse=True):
    test = np.delete(test, i, 0) #delete correctly classified from test data

##for printing purposes only
NBpredicted = list(zip(NBpredLabel, NBtestLabels, NBtestFeatures))
pd.DataFrame(NBpredicted).reset_index().to_csv('nbcresults.csv', index=False, header = [' ', 'predicted', 'label',  
'article']) #results of NBC

##for printing purposes only
newTest = list(zip(l_testAgain, f_testAgain))
pd.DataFrame(newTest).reset_index().to_csv('testfordt.csv', index=False, header = [' ', 'label',  'article'])

###APPLYING CLASSIFIERS (2)
## DECISION TREE ALGORITHM
fromNBfeatures = pd.DataFrame(f_testAgain) #for testing
fromNBlabels = pd.DataFrame(l_testAgain) #for testing

dt_startTime = time.time()

DTmodel = DecisionTreeClassifier().fit(train, trainLabels) ##train the model
DTprediction = DTmodel.predict(test) ##making predictions
#DTprediction = DTmodel.predict([test])

##put back classified data in same index 
j = 1
for i in sorted(misclassified, reverse=False): #put every DT classified back
    NBprediction = np.insert(NBprediction, i, DTprediction[len(DTprediction)-j]) #add DT classified data back 
    j+=1

##calculate elapsed time
dt_endTime = time.time()
nbdt_elapsedTime = (dt_endTime - dt_startTime)+nb_elapsedTime

###EVALUATION (2)
##evalute naive bayes - decision tree
print('\nNAIVE BAYES - DECISION TREE ALGORITHM')
print('Test Size:', testSize)
print('Accuracy:', accuracy_score(testLabels, NBprediction))
print('F1 score:', f1_score(testLabels, NBprediction, average="macro"))
print('Runtime:', nbdt_elapsedTime)

##for printing purposes only
DTpredLabel = [item for item in DTprediction]
finalPrediction = list(zip(DTpredLabel, l_testAgain, f_testAgain))
pd.DataFrame(finalPrediction).reset_index().to_csv('dtresults.csv', index=False, header = [' ', 'predicted',
'label', 'article'])