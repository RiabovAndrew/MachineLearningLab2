# MachineLearningLab2<br>Andrey Ryabov

![image](https://user-images.githubusercontent.com/43186510/212141774-191643a3-db0c-4ede-bc2f-5564778e1f08.png)

## Solution

1) Let's create **Crisp Sets** with **Terms**. Also add **deffuzification** multipliers for every **Crisp Set**.
```
Temperature = ["Cold", "Medium", "Hot"]         # degree
TemperatureMultiplier = [3, 1, 3]

Rain = ["Drizzling", "Raining", "Showering"]    # drops/s
RainMultiplier = [1.1, 2, 3]

WindSpeed = ["Quite", "Feels", "BlowsAway"]     # m/s
WindMultiplier = [1, 2, 5]

Path = ["Easy", "Medium", "Hard"]               # multiplier
```
2) Now write the function which prepares coefficient weights for part of linear function.
```
def getWeight(x, left, right, inverted=False):
    if x <= left: return 1 if inverted else 0
    if x >= right: return 0 if inverted else 1
    if inverted:
        return 1 - (x - left) / (right - left)
    return (x - left) / (right - left)
```
3) Here create the function which prepares coefficient weights for part of linear function, but triangulary shaped.
```
def getTriangleWeight(x, left, center, right, inverted=False):
    return getWeight(x, left, center, inverted) if x <= center else getWeight(x, center, right, not inverted)
```
4) Temperature, Rain, WindSpeed **Fuzzy Sets**.
```
def tempFuncWeights(x):
    return (getWeight(x, 0, 10, True), getTriangleWeight(x, 7, 15, 27), getWeight(x, 27, 31))

def rainFuncWeights(x):
    return (getTriangleWeight(x, 0, 1, 2), getWeight(x, 0, 4), getWeight(x, 0, 7))

def windFuncWeights(x):
    return (getWeight(x, 0, 2, True), getTriangleWeight(x, 2, 4, 7), getWeight(x, 7, 15))
```
5) A simple function for _First of Maximum_ method calculation.
```
def FoM(tuple):                # First of Maximum
    maxValue = max(tuple)
    if tuple[0] == maxValue:
        return (maxValue, 0)
    if tuple[1] == maxValue:
        return (maxValue, 1)
    return (maxValue, 2)
```
6) Finally, write the defuzzification function using _FoM (First of Maximum)_ method. Also this defuzzification function works with **Mamdani inferencing** shown in p.1.
```
def calcDefuzzificatedValue(w1, w2, w3):
    res1 = FoM(w1)[0] * TemperatureMultiplier[FoM(w1)[1]]
    res2 = FoM(w2)[0] * RainMultiplier[FoM(w2)[1]]
    res3 = FoM(w3)[0] * WindMultiplier[FoM(w3)[1]]

    return res1 + res2 + res3
```
7) The other code just shows how my implementation works.
```
xses = (10, 5, 15)
print("How hard to go on a trip with (Temperature =", xses[0], ", Rain =", xses[1], " WindSpeed =", xses[2], ") ==", 
    calcDefuzzificatedValue(tempFuncWeights(xses[0]), rainFuncWeights(xses[1]), windFuncWeights(xses[2])))

x = np.linspace(0, 31, 10)
y = np.linspace(0, 7, 10)
windSpeed = 15

X, Y = np.meshgrid(x, y)

Z = [[calcDefuzzificatedValue(tempFuncWeights(i), rainFuncWeights(j), windFuncWeights(windSpeed)) for i in x] for j in y]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("Temperature, degrees")
ax.set_ylabel("Rain, drops/s")
ax.set_zlabel("How hard to go on a trip with {}, m/s wind speed".format(windSpeed))
ax.contour3D(X, Y, Z, 100, cmap='viridis')
plt.show()
```
## Results
1) An output text.

![image](https://user-images.githubusercontent.com/43186510/212159049-63c62a2b-90ec-41dc-a658-bfba81171c0f.png)

2) Images of my **Mamdani inferencing**.

![image](https://user-images.githubusercontent.com/43186510/212159336-04b1c5cd-9e93-4fad-9022-d16348822571.png)

or from another side

![image](https://user-images.githubusercontent.com/43186510/212159650-77c1be0c-49e6-4311-b8d8-ff2d0357e958.png)

## Code of solution
main.py
```
import numpy as np
import matplotlib.pyplot as plt

Temperature = ["Cold", "Medium", "Hot"]         # degree
TemperatureMultiplier = [3, 1, 3]

Rain = ["Drizzling", "Raining", "Showering"]    # drops/s
RainMultiplier = [1.1, 2, 3]

WindSpeed = ["Quite", "Feels", "BlowsAway"]     # m/s
WindMultiplier = [1, 2, 5]

Path = ["Easy", "Medium", "Hard"]               # multiplier

def getWeight(x, left, right, inverted=False):
    if x <= left: return 1 if inverted else 0
    if x >= right: return 0 if inverted else 1
    if inverted:
        return 1 - (x - left) / (right - left)
    return (x - left) / (right - left)

def getTriangleWeight(x, left, center, right, inverted=False):
    return getWeight(x, left, center, inverted) if x <= center else getWeight(x, center, right, not inverted)

def tempFuncWeights(x):
    return (getWeight(x, 0, 10, True), getTriangleWeight(x, 7, 15, 27), getWeight(x, 27, 31))

def rainFuncWeights(x):
    return (getTriangleWeight(x, 0, 1, 2), getWeight(x, 0, 4), getWeight(x, 0, 7))

def windFuncWeights(x):
    return (getWeight(x, 0, 2, True), getTriangleWeight(x, 2, 4, 7), getWeight(x, 7, 15))

def calcDefuzzificatedValue(w1, w2, w3):
    res1 = FoM(w1)[0] * TemperatureMultiplier[FoM(w1)[1]]
    res2 = FoM(w2)[0] * RainMultiplier[FoM(w2)[1]]
    res3 = FoM(w3)[0] * WindMultiplier[FoM(w3)[1]]

    return res1 + res2 + res3

def FoM(tuple):                # First of Maximum
    maxValue = max(tuple)
    if tuple[0] == maxValue:
        return (maxValue, 0)
    if tuple[1] == maxValue:
        return (maxValue, 1)
    return (maxValue, 2)

xses = (10, 5, 15)
print("How hard to go on a trip with (Temperature =", xses[0], ", Rain =", xses[1], " WindSpeed =", xses[2], ") ==", 
    calcDefuzzificatedValue(tempFuncWeights(xses[0]), rainFuncWeights(xses[1]), windFuncWeights(xses[2])))

x = np.linspace(0, 31, 10)
y = np.linspace(0, 7, 10)
windSpeed = 15

X, Y = np.meshgrid(x, y)

Z = [[calcDefuzzificatedValue(tempFuncWeights(i), rainFuncWeights(j), windFuncWeights(windSpeed)) for i in x] for j in y]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("Temperature, degrees")
ax.set_ylabel("Rain, drops/s")
ax.set_zlabel("How hard to go on a trip with {}, m/s wind speed".format(windSpeed))
ax.contour3D(X, Y, Z, 100, cmap='viridis')
plt.show()
```
