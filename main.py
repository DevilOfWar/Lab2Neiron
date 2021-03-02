import numpy as np


def transform(x, n):
    if x < n:
        return 0
    else:
        return 1


def f(x):
    return 1 / (1 + np.exp(-x))

def nonlin(x,deriv=False):
	if(deriv==True):
           return f(x)*(1-f(x))
	return f(x)


inputData = [[0, 0, 0, 0, 0, 0]]
outputData = [0]
n = [1.0, 4.0, 30.0, 4.0, 12.0, 1.0]
flag = False
with open("tests1.txt", "r") as file:
    for line in file:
        list = line.split(" ")
        if flag is True:
            inputData.append([0, 0, 0, 0, 0, 0])
        index = 0
        for element in list:
            if element != "\n":
                element_float = float(element)
                if index <= 5:
                    inputData[-1][index] = transform(element_float, n[index])
                else:
                    if flag is False:
                        outputData[-1] = element_float
                    else:
                        outputData.append(element_float)
                index = index + 1
        if flag is False:
            flag = True

inputData = np.array(inputData)
outputData = np.array(outputData)
weight1 = np.random.random((6,2))
weight2 = np.random.random((2,1))

print(f"До обучения:")
print(f"{weight1.T} - весы первого уровня")
print(f"{weight2.T} - весы второго уровня")

AnsUnlearned = nonlin(np.dot(nonlin(np.dot(inputData, weight1)), weight2))

learnTests = [[0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 1, 0],
              [0, 1, 1, 0, 1, 1],
              [0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 1],
              [0, 1, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 1],
              [1, 0, 0, 1, 0, 1],
              [1, 0, 0, 1, 1, 0],
              [1, 0, 0, 1, 1, 1],
              [1, 0, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1, 1],
              [1, 0, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 1],
              [1, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 1],
              [1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 1],
              [1, 1, 0, 0, 1, 0],
              [1, 1, 0, 0, 1, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 0, 1, 0, 1],
              [1, 1, 0, 1, 1, 0],
              [1, 1, 0, 1, 1, 1],
              [1, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 1, 0],
              [1, 1, 1, 0, 1, 1],
              [1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 1]]
#learnTestsAns = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
learnTestsAns = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
learnTestsAns = np.array(learnTestsAns)
learnTests = np.array(learnTests)
learnSpeed = 0.2

for j in range(1000):
    layer1 = learnTests
    layer2 = nonlin(np.dot(layer1, weight1))
    layer3 = nonlin(np.dot(layer2, weight2)).T
    delta = 0
    for i in range(0, len(learnTestsAns)):
        delta += learnTestsAns[i] - layer3.T[i]
    delta = delta / len(learnTestsAns)
    print(f"{j} итерация обучения. Средняя погрешность:{delta}.")
    layerError3 = np.subtract(learnTestsAns, layer3)
    layerDelta3 = np.multiply(layerError3, nonlin(layer3, True))
    layerError2 = layerDelta3.T.dot(weight2.T)
    layerDelta2 = np.multiply(layerError2, nonlin(layer2, True))
    weight2 = np.add(weight2, layer2.T.dot(layerDelta3.T) * learnSpeed)
    weight1 = np.add(weight1, np.array(layer1).T.dot(layerDelta2) * learnSpeed)
    print(f"{weight1.T} - весы первого уровня")
    print(f"{weight2.T} - весы второго уровня")

AnsLearned = nonlin(np.dot(nonlin(np.dot(inputData, weight1)), weight2))

deltaUnlearned = 0
deltaLearned = 0
for j in range(0, len(inputData)):
    print(f"{j} тест.\nПреобразованные входные данные:{inputData[j]}.\nОжидаемый выход:{outputData[j]}.\nДо обучения:{AnsUnlearned[j]} -> {round(AnsUnlearned[j][0])}.\nПосле обучения:{AnsLearned[j]} -> {round(AnsLearned[j][0])}.\n")
    deltaUnlearned += abs(outputData[j] - round(AnsUnlearned[j][0]))
    deltaLearned += abs(outputData[j] - round(AnsLearned[j][0]))
deltaUnlearned = deltaUnlearned / len(outputData)
deltaLearned = deltaLearned / len(outputData)
print(f"Средняя погрешность до обучения:{deltaUnlearned}; после обучения:{deltaLearned}.")
