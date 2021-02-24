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
n = [1, 4, 30, 4, 12, 1]
flag = False
with open("tests.txt", "r") as file:
    for line in file:
        list = line.split(" ")
        if flag is True:
            inputData.append([0, 0, 0, 0, 0, 0])
        index = 0
        for element in list:
            if element != "\n":
                element_float = float(element)
                if index < 5:
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
weight1 = 2 * np.random.random((6,2)) - 1
weight2 = 2 * np.random.random((2,1)) - 1
learnTests = []
learnTestsAns = []
for j in range(100):
    i = np.random.randint(0, len(inputData))
    learnTests.append(inputData[i])
    learnTestsAns.append(outputData[i])

AnsUnlearned = nonlin(np.dot(nonlin(np.dot(inputData, weight1)), weight2))

for j in range(1000):
    layer1 = learnTests
    layer2 = nonlin(np.dot(layer1, weight1))
    layer3 = nonlin(np.dot(layer2, weight2)).T
    layerError3 = np.subtract(learnTestsAns, layer3)
    layerDelta3 = np.multiply(layerError3, nonlin(layer3, True))
    layerError2 = layerDelta3.T.dot(weight2.T)
    layerDelta2 = np.multiply(layerError2, nonlin(layer2, True))
    weight2 = np.add(weight2, layer2.T.dot(layerDelta3.T))
    weight1 = np.add(weight1, np.array(layer1).T.dot(layerDelta2))

AnsLearned = nonlin(np.dot(nonlin(np.dot(inputData, weight1)), weight2))
deltaUnlearned = 0
deltaLearned = 0
for j in range(0, len(inputData)):
    print(f"{j} тест.\nВходные данные:{inputData[j]}.\nОжидаемый выход:{outputData[j]}.\nДо обучения:{AnsUnlearned[j]}.\nПосле обучения:{AnsLearned[j]}\n")
    print(f"Погрешность до обучения:{outputData[j] - AnsUnlearned[j]}.\nПогрешность после обучения:{outputData[j] - AnsLearned[j]}.\n")
    deltaUnlearned += abs(outputData[j] - AnsUnlearned[j])
    deltaLearned += abs(outputData[j] - AnsLearned[j])
deltaUnlearned = deltaUnlearned / 1000.0
deltaLearned = deltaLearned / 1000.0
print(f"Средняя погрешность до обучения:{deltaUnlearned}; после обучения:{deltaLearned}.")
