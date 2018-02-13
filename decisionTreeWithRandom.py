# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:43:27 2018

@author: Tommy
"""
from sklearn.model_selection import train_test_split
import math 
import random
import operator
import pandas

""" 
Get attributeName from dataFiles
""" 
def getAttributes(dataSetName):
    attributes=[]
    file = open(dataSetName,'r')
        
    n = file.readline()
    datas = n.split(',')

    for i in datas:
        i = i.replace('\n','')
        attributes.append(i)
    file.close()
    return attributes

""" 
Get example dataSets from dataFiles
"""      
def createDataSet(dataSetName): 
        
    dataSet = []
    attributes=[]
    file = open(dataSetName,'r')
    
    n = file.readline()
    datas = n.split(',')

    for i in datas:
        i = i.replace('\n','')
        attributes.append(i) 
        
    # ================= data ===================
    n = file.readline()
    temp = []
    while n :       
        n = n.split(',')
        for i in n:
            i = i.replace('\n','')
            i = i.replace(' ','')
            temp.append(i)
        #print(len(temp),'==',temp)
        dataSet.append(temp)
        n = file.readline()
        temp=[]   
    file.close()
    '''
    dataSet = [ ['Sunny', 'Hot','Hight', 'Weak', 'no'],  
                ['Sunny', 'Hot','Hight', 'Strong', 'no'],  
                ['Overcast', 'Hot','Hight', 'Weak', 'yes'],  
                ['Rain', 'Mild','Hight', 'Weak', 'yes'],
                ['Rain', 'Cool','Normal', 'Weak', 'yes'],
                ['Rain', 'Cool','Normal', 'Strong', 'no'],
                ['Overcast', 'Cool','Normal', 'Strong', 'yes'],
                ['Sunny', 'Mild','Hight', 'Weak', 'no'],
                ['Sunny', 'Cool','Normal', 'Weak', 'yes'],
                ['Rain', 'Mild','Normal', 'Weak', 'yes'],
                ['Sunny', 'Mild','Normal', 'Strong', 'yes'],
                ['Overcast', 'Mild','Hight', 'Strong', 'yes'],
                ['Overcast', 'Hot','Normal', 'Weak', 'yes'],
                ['Rain', 'Mild','Hight', 'Strong', 'no']
                ]      
    # debug tool
    for i,n in enumerate(dataSet):
        if n!=dataSet2[i]:
            print(i)
            for jind,j in enumerate(n):
                if j != dataSet2[i][jind]:
                    print('this fasle = ',jind)
    '''
    return dataSet

""" 
Dataset splitting on a given feature  
dataSet : original data set  
index : cut Feature's column index.  
value : cut Feature's value.
return a dataSet after splite the given feature
""" 
def reduceDataSet(dataSet, index, value):  
 
    reducedDataSet = []  
    for instance in dataSet:
        if instance[index] == value:
            reducedFeattureVec = instance[:index]            
            reducedFeattureVec.extend(instance[index+1:])  
            reducedDataSet.append(reducedFeattureVec)
    return reducedDataSet

""" 
Calculate the Entropy of a dataSet
Entropy(S) = −p+ log(p+) − p− log(p−)
"""   
def calculateEntropy(dataSet):
 
    numOfExample = len(dataSet)
    classCounts = {} # store each Key:label has how many Value:counts
    for example in dataSet:
        currentLabel = example[-1] 
        classCounts[currentLabel] = classCounts.get(currentLabel, 0) + 1 # count each classify class
    Entropy = 0.0
    #print(labelCounts)
    for label in classCounts:  
        prob = float(classCounts[label]) / numOfExample
        Entropy = Entropy - prob * math.log2(prob)

    return Entropy

""" 
Calculate Gain 
with handle missing data
One way to handle missing values is to assign the most likely value of xi to s
""" 
def calculateGain(dataSet, index, baseEntropy):

    featureList = []# total features in each column
    for example in dataSet:
        featureList.append(example[index])
    ####handle missing data : One way to handle missing values is to assign the most likely value of xi to s
    featListWithoutMissing = []
    for example in dataSet:
        if(example[index] == '?'):
            example[index] = countMostCommonValue(featureList)
                
        featListWithoutMissing.append(example[index])
    #print('Most Common Feature=', countMostCommonValue(featureList))
    #print('Original List=', featureList)
    #print('Covered  List=', featListWithoutMissing)
    #####################
    removeDuplicateValues = set(featListWithoutMissing) # identify how many different features in the featurelist
    newEntropy = 0
    for value in removeDuplicateValues:  
        subDataSet = reduceDataSet(dataSet, index, value)  
        prob = len(subDataSet) / float(len(dataSet))
        newEntropy = newEntropy + (prob * calculateEntropy(subDataSet))  
    gain = baseEntropy - newEntropy  # Information gain: after Split, data become more organized, then Entropy will decrease.  
    return gain

""" 
Choosing the best feature to split on
dataSet: original data set  
return: the Feature column index which has max Information gain.
""" 
def calculateBestFeature(dataSet):  
 
    baseEntropy = calculateEntropy(dataSet) ## calculate the Entropy(S)
    numOfFeatures = len(dataSet[0]) - 1 # last Feature is classify Label
    #print('baseEntropy: ',baseEntropy)
    bestGain = 0.0 
    featureIndex = -1  
    for i in range(numOfFeatures):
        infoGain = calculateGain(dataSet, i, baseEntropy)
        if infoGain > bestGain:  
            bestGain = infoGain  
            featureIndex = i
    return featureIndex 

""" 
Use Majority Vote to choose most common feature and return.  
featureList: a column of attribute   
return: the most common class in the List.
"""
def countMostCommonValue(featureList):  

    countResult = {}  
    for feature in featureList:  
        countResult[feature] = countResult.get(feature, 0) + 1  
    countResultSorted = sorted(countResult.items(), key = operator.itemgetter(1), reverse=True)
    if(countResultSorted[0][0] == '?'):
        return countResultSorted[1][0] #if the most common feature is '?', then return the second common feature as the most common feature
    return countResultSorted[0][0]

""" 
Tree-building code. 
dataSet: original dataset  
attributes: attribute names
return: Decision Tree.
"""
def buildDecisionTree(dataSet, attributes):  
 
    classList = []#catch the classes
    for instance in dataSet:
        classList.append(instance[-1])
    if classList.count(classList[0]) == len(classList):     # if all int the same class then add classList[0][0] as leaf node
        #print(classList[0])
        return classList[0]
    if len(dataSet[0]) == 1:                                # if instance is Empty then add leaf node with label = most common value of target_attribute in Example  
        return countMostCommonValue(classList)  
    bestFeature = calculateBestFeature(dataSet) 
    bestFeatAttribute = attributes[bestFeature]  
    decisionTree = {bestFeatAttribute:{}}  
    del(attributes[bestFeature])  
    featValues = [instance[bestFeature] for instance in dataSet] 
    uniqueVals = set(featValues)  
    for value in uniqueVals:  
        subAttributes = attributes[:]
        #print(subAttributes)
        newReduceDataSet = reduceDataSet(dataSet, bestFeature, value)
        decisionTree[bestFeatAttribute][value] = buildDecisionTree(newReduceDataSet, subAttributes)
    return decisionTree  

""" 
Classification function for an existing decision tree
"""
def predict(inputTree, featureAttributes, testData):

    firstStr = list(inputTree.keys())[0]                #catch Root node
    nextDict = inputTree[firstStr]                    #catch currentRoot's decisionTree.
    featIndex = featureAttributes.index(firstStr)       #catch feature's attribute name.
    classLabel = 'NotExist'
    for key in nextDict.keys():
        if testData[featIndex] == key:                               
            if type(nextDict[key]).__name__ == 'dict':            # still tree keepWalk
                classLabel = predict(nextDict[key], featureAttributes, testData)
                #print("------------------",classLabel)
            else:                                                   #if not tree return value
                #print('---------------',secondDict[key])
                return nextDict[key]                            # if feature value's branch is class, return.  
    return classLabel

""" 
Calculate Accuracy of dataSet
"""
def calculateAccuracy(dataSet,tree, attributes):    # calculate the accuracy of training data and testing data
    dataSetLen = len(dataSet)
    numCorrectData = 0
    dataLen = len(dataSet)
    
    for i in range(dataLen):
        result = predict(tree, attributes, dataSet[i])
        targetResult =  dataSet[i][-1]
        if result == targetResult:
            numCorrectData = numCorrectData + 1

    #print('Total number of correct TestData= ', numCorrectData)
    #print('Total number of TestData= ', dataSetLen)    
    correctRate = numCorrectData / dataSetLen
    #print('Testing Data Accuracy= ',correctRate * 100,'%')
    #print('============================================')
    return numCorrectData,dataSetLen,correctRate
            
def doProgram(filename, numOfRandomForest):
    totalTestAccuracy = 0
    totalTrainAccuracy = 0
    
    dataSet= createDataSet(filename)
    dataSetLen = len(dataSet)
    #print(dataSet)
    for i in range(numOfRandomForest):
        random.shuffle(dataSet)

        rateOfTestDataInDataSet = 0.2
        trainData, testData = train_test_split(dataSet, test_size=rateOfTestDataInDataSet) #seperate data into train 0.8 test 0.2
        
        attributes = getAttributes(filename)
        tree = buildDecisionTree(trainData, attributes)            #create tree
        attributes = getAttributes(filename)
        attributesLen = len(attributes)
        
        print('Tree',i+1,'================Decision Tree==========================')
        #print('Decision Tree= ', tree)
        print('')
        print('Total number of DataSet=',dataSetLen)
        print('')
        print('Total number of Attributes=', attributesLen)
        #print('Attributes=',attributes)
        print('Tree',i+1,'================ TestData Accuracy=====================')
        numCorrectTestData, numTestData, testAccuracy = calculateAccuracy(testData,tree, attributes)
        print('Total number of correct TestData= ', numCorrectTestData)
        print('Total number of TestData= ', numTestData, ',Percentage of Total DataSet =',rateOfTestDataInDataSet*100,'%')    
        print('Testing Data Accuracy= ',testAccuracy*100,'%')
        totalTestAccuracy = totalTestAccuracy + testAccuracy;
        print('')
        
        print('Tree',i+1,'================ TrainData Accuracy====================')
        numCorrectTrainData, numTrainData, tainAccuracy = calculateAccuracy(trainData,tree, attributes)
        print('Total number of correct TrainData= ', numCorrectTrainData)
        print('Total number of TrainData= ', numTrainData, ',Percentage of Total DataSet =',(1-rateOfTestDataInDataSet)*100,'%')
        print('Training Data Accuracy= ',tainAccuracy*100,'%')
        totalTrainAccuracy = totalTrainAccuracy + tainAccuracy;
        print('')
    
    print('Total number of Decision Tree=',numOfRandomForest);
    accuTest = totalTestAccuracy/numOfRandomForest;
    accuTrain = totalTrainAccuracy/numOfRandomForest;
    print('AVGTesting Data Accuracy= ',accuTest*100,'%')
    print('AVGTraining Data Accuracy= ',accuTrain*100,'%')
           
def main():   
    #filename = 'playTennis.txt'
    #filename = 'carEvaluation.txt'
    #filename = 'VotingRecords.txt'
    filename = 'mushroom.txt'
    numOfRandomForest = 10
    doProgram(filename, numOfRandomForest)
    
    
main()
