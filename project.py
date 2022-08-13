import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import csv

plt.style.use('ggplot')

mutation = True
elitism = False
mutationPorcentage = .4
tournamentPercentage = 0.02
generations = 100
nPopulation = 100


Nu_mfx = 3
Nu_mfy = 3

chromosomeSize = (Nu_mfx*2) + (Nu_mfy*2) + (Nu_mfx*Nu_mfy*3)

size = 1000

xValues = [i for i in range(1,21)]
yValues = [i for i in range(1,32)]
sizeX = 20
sizeY = 32

population = []

fa = []

main3Dgraph = []
plotDistanceX = []
plotDistanceY = []

dataInegi = []

line = []

def creatMain3DGraph():
    global main3Dgraph
    global dataInegi

    yDataArray = []
    xDataArray = []

    with open('data2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                xDataArray = [int(i) for i in row[1:]]
                line_count += 1
            else:
                yDataArray.append(int(row[0]))
                dataInegi.append([int(i) for i in row[1:]])
                line_count += 1

    X, Y = np.meshgrid(xDataArray, yDataArray)

    main3Dgraph =[X, Y, np.array(dataInegi)]
    

def make3dGraph(data):

    Y, X = np.meshgrid(data[1], data[0])

    Z = np.array(data[2])

    return [X, Y, Z]

def createPopulation():
    for i in range(0, nPopulation):
        chromosome=random.sample(range(0,256), chromosomeSize)
        population.append(chromosome)
    
def getFA(po):
    
    poFA = []
    for chromosome in po:
        fa_v = 0
        for x in xValues:
            for y in yValues:
                
                z, mfx, mfy = getZFuzzy(chromosome, x, y)

                ref = dataInegi[y-1][x-1]

                fa_v += abs(ref - z)

        poFA.append(fa_v)
    return poFA

def populateFA(ei):
    fa.clear()

    sDistance = 999999999999999999999999
    plotBest = []

    for chromosome in population:
        fa_v = 0
        plotGraphZTemp = []
        plotGraphMfxTemp = []
        plotGraphMfyTemp = []

        for x in xValues:
            temp = []
            for y in yValues:
                
                z, mfx, mfy = getZFuzzy(chromosome, x, y)

                if(len(plotGraphMfyTemp) < len(yValues)):
                    plotGraphMfyTemp.append(mfy)
                
                temp.append(z)
            
                ref = dataInegi[y-1][x-1]

                fa_v += abs(ref - z)
            
            plotGraphMfxTemp.append(mfx)

            plotGraphZTemp.append(temp)
        fa.append(fa_v)


        if sDistance > fa_v:
            sDistance = fa_v
            plotBest = [xValues, yValues, plotGraphZTemp]
    
    plotDistanceX.append(ei)
    plotDistanceY.append(sDistance)

    distanceGraph = [plotDistanceX, plotDistanceY]


    mfxGraph = make2dGraph(xValues, plotGraphMfxTemp)
    mfyGraph = make2dGraph(yValues, plotGraphMfyTemp)

    live_plotter(distanceGraph, mfxGraph, mfyGraph, make3dGraph(plotBest), sDistance)

def make2dGraph(x, yArray):
    graph = []
    for i in range(0, len(yArray[0])):
        temp = []
        for e in yArray:
            temp.append(e[i])
        graph.append([x,temp])

    return graph
        
def betterOptions(participants):
    betterOption = participants[0]
    for i in participants:
        if(fa[betterOption] > fa[i]):
            betterOption = i

    return betterOption

def reproduction(f,m):
    cutPoint = random.randint(0, chromosomeSize * 8)
    cutIndex = int(cutPoint / 8)

    n = cutPoint - (cutIndex * 8)

    if(n == 0):
        fc = f[0:cutIndex] + m[cutIndex:chromosomeSize]
        mc = m[0:cutIndex] + f[cutIndex:chromosomeSize]
    else:

        lowMask = (2**n)-1
        highMask = ((2**chromosomeSize)-1)-((2**n)-1)

        lowPar1 = f[cutIndex] & lowMask
        lowPar2 = m[cutIndex] & lowMask

        highPar1 = f[cutIndex] & highMask
        highPar2 = m[cutIndex] & highMask

        child1 = lowPar1 | highPar2
        child2 = lowPar2 | highPar1

        fc = f[0:cutIndex] + [child2] + m[cutIndex+1:chromosomeSize]
        mc = m[0:cutIndex] + [child1] + f[cutIndex+1:chromosomeSize]

    return fc,mc

def mutation(childList):
    participants = random.sample(range(0,nPopulation), int(nPopulation * mutationPorcentage))
    
    for i in participants:
        cutPoint = random.randint(0, (chromosomeSize * 8) - 1)

        numberIndex = int(cutPoint / 8)

        n = cutPoint - (numberIndex * 8)

        numberToMutate = childList[i][numberIndex]

        bindata = '{0:08b}'.format(numberToMutate)

        bitNot = '1' if bindata[n] == '0' else '0'
        new = bindata[0:n] + bitNot + bindata[n+1:8]
        
        newInt = int(new, 2)

        childList[i][numberIndex] = newInt


def tournament():
    global population

    childList = []
    for i in range(0, int(nPopulation/2)):
            participants = random.sample(range(0,nPopulation), int(nPopulation * tournamentPercentage))
            f = betterOptions(participants)

            participants = random.sample(range(0,nPopulation), int(nPopulation * tournamentPercentage))
            m = betterOptions(participants)

            a,b = reproduction(population[f],population[m])

            childList.append(a)
            childList.append(b)
    
    mutation(childList)

    if(elitism):
        completeList = population + childList
        faList = fa + getFA(childList)

        listIndex = {v: k for v, k in enumerate(faList)}
        sortedListIndex = list(dict(sorted(listIndex.items(), key=lambda item: item[1])).items())
        populationIndex = sortedListIndex[0:nPopulation]
        
        betterList = [v for v, k in populationIndex]

        population = [completeList[v] for v in betterList]
        
    else:
        population = childList



def live_plotter(attitude_graph, mfx_graph, mfy_graph, graph3d, sDistance,pause_time=0.01):
    global ax1
    global ax2
    global ax3
    global ax0
    global line
    global main3Dgraph

    global plt

    if line==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()

        fig0 = plt.figure(figsize=(10,6))
        fig0.canvas.manager.window.wm_geometry("+0+0")
        ax0 = fig0.add_subplot(111, projection="3d")


        fig1 = plt.figure(figsize=(8.5,6))
        fig1.canvas.manager.window.wm_geometry("+1200+0")
        ax1 = fig1.add_subplot(111)
        line, = ax1.plot(attitude_graph[0],attitude_graph[1],'-o',alpha=0.8)      

        

        fig2 = plt.figure(figsize=(9,4))
        fig2.canvas.manager.window.wm_geometry("+0+1000")
        ax2 = fig2.add_subplot(111)

        fig3 = plt.figure(figsize=(9,4))
        fig3.canvas.manager.window.wm_geometry("+1100+1000")
        ax3 = fig3.add_subplot(111)

    plt.title( str(sDistance))
    
    line.set_xdata(attitude_graph[0])
    line.set_ydata(attitude_graph[1])


    ax2.cla()
    for graph in mfx_graph:
        ax2.plot(graph[0],graph[1],'-o',alpha=0.8)
    ax2.set_title('mfx')

    ax3.cla()
    for graph in mfy_graph:
        ax3.plot(graph[0],graph[1],'-o',alpha=0.8)

    ax3.set_title('mfy')


    ax0.cla()
    ax0.plot_surface(graph3d[0], graph3d[1], graph3d[2], color='blue')
    ax0.plot_wireframe(main3Dgraph[0], main3Dgraph[1], main3Dgraph[2])

    labelsx = ['Menores de 1', '1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', 'mas de 85', 'No especificado']
    labelsy =[str(i) for i in range(1990,2022)]
    ax0.set_xticklabels(labelsx)
    ax0.set_yticklabels(labelsy)


    # adjust limits if new data goes beyond bounds
    if np.min(attitude_graph[1])<=line.axes.get_ylim()[0] or np.max(attitude_graph[1])>=line.axes.get_ylim()[1]:
        ax1.set_ylim([np.min(attitude_graph[1])-np.std(attitude_graph[1]),np.max(attitude_graph[1])+np.std(attitude_graph[1])])

      # adjust limits if new data goes beyond bounds
    if np.min(attitude_graph[0])<=line.axes.get_xlim()[0] or np.max(attitude_graph[0])>=line.axes.get_xlim()[1]:
        ax1.set_xlim([np.min(attitude_graph[0])-np.std(attitude_graph[0]),np.max(attitude_graph[0])+np.std(attitude_graph[0])])


    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

def getZFuzzy(data_fuzzy, x,y):

    x = x/10
    y = y/10
    data_fuzzyt = []
    for i in range(0, len(data_fuzzy)):

        if(i < Nu_mfx+Nu_mfy):
            data_fuzzyt.append(data_fuzzy[i]/100)
        elif(i < (Nu_mfx+Nu_mfy)*2):
            data_fuzzyt.append(data_fuzzy[i]/100)
        else:
            data_fuzzyt.append(data_fuzzy[i]*100)

    
    # data_fuzzyt = [i*10 for i in data_fuzzy]

    d_start = Nu_mfx+Nu_mfy
    mfx = []
    for e in range(0, Nu_mfx):
        if (data_fuzzyt[e+d_start] == 0):
            mfx.append(0)
        else:
            mfx.append(math.exp((-(x-data_fuzzyt[e])**2)/(2*data_fuzzyt[e+d_start]**2)))

    mfy = []
    for e in range(0, Nu_mfy):
        if (data_fuzzyt[e+d_start+Nu_mfx] == 0):
            mfy.append(0)
        else:
            mfy.append(math.exp((-(y-data_fuzzyt[e+Nu_mfx])**2)/(2*data_fuzzyt[e+d_start+Nu_mfx]**2)))

    inf = []
    for e in range(0, Nu_mfx):
        for u in range(0, Nu_mfy):
            inf.append(mfx[e]*mfy[u])
        
    p_start = (Nu_mfx+Nu_mfy)*2
    q_start = p_start + (Nu_mfx*Nu_mfy)
    r_start = q_start + (Nu_mfx*Nu_mfy)

    reg = []
    for e in range(0, Nu_mfx*Nu_mfy):
        reg.append(inf[e]*((data_fuzzyt[e+p_start]*x)+(data_fuzzyt[e+q_start]*y)+data_fuzzyt[e+r_start]))

    a = 0
    b = 0

    for e in range(0, Nu_mfx*Nu_mfy):
        b += inf[e]
        a += reg[e]
    if(b == 0):
        z = 0
    else:
        z = a/b
    return z, mfx, mfy

if __name__ == "__main__":
    
    creatMain3DGraph()
    createPopulation()

    for i in range(0, generations):
        populateFA(i)

        tournament()
    input("Press Enter to continue...")
