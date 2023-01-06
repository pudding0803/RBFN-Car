import math
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

from Neuron import Neuron


def line_to_coord(line: str) -> list:
    if line[-1] == '\n':
        line = line[:-1]
    return list(map(int, line.split(',')))


def distances(car: list, deg: float) -> np.ndarray:
    xs = [(-6, -3, 22), (6, -3, 10), (18, 22, 50), (30, 10, 50)]
    ys = [(-3, -6, 6), (10, 6, 30), (22, -6, 18), (50, 18, 30)]
    dists = [(0, 0, math.inf) for _ in range(3)]
    for i in range(3):
        for x, low, high in xs:
            try:
                k = (x - car[0]) / math.cos(deg + (i - 1) * math.pi / 4)
                y = car[1] + k * math.sin(deg + (i - 1) * math.pi / 4)
                dist = math.dist(car, (x, y))
                if k >= 0 and low <= y <= high and dist < dists[i][2]:
                    dists[i] = (x, y, dist)
            except ZeroDivisionError:
                continue
        for y, low, high in ys:
            try:
                k = (y - car[1]) / math.sin(deg + (i - 1) * math.pi / 4)
                x = car[0] + k * math.cos(deg + (i - 1) * math.pi / 4)
                dist = math.dist(car, (x, y))
                if k >= 0 and low <= x <= high and dist < dists[i][2]:
                    dists[i] = (x, y, dist)
            except ZeroDivisionError:
                continue
    return np.array([dists[1], dists[2], dists[0]])


class RBFN:
    def __init__(self):
        while True:
            try:
                self.d = int(input('Input 4 or 6 for the train data (dimension): '))
                if self.d == 4 or self.d == 6:
                    break
            except ValueError:
                pass
        path = f'./train{self.d}dAll.txt'
        if not os.path.exists(path):
            print(f'{path} does not exist.')
            print('Please check if the file is in the same directory.')
            os.system('pause')
            return
        with open(path) as file:
            line = file.readline().split()
            self.data = np.array([[float(i) for i in line] + [math.inf, -1]])
            for line in file:
                self.data = np.append(self.data, [[float(i) for i in line.split()] + [math.inf, -1]], axis=0)
        self.d -= 1
        self.neurons = []
        print('Doing K-means... It won\'t take too much time.')
        self.kmeans(15)
        print(f'There are {len(self.neurons)} neurons in the hidden layer.\n')

        epoch, bias = 50, -1
        for e in range(epoch):
            loss = 0
            eta = (epoch - e) / epoch * 0.1
            for data in self.data:
                output = bias
                for neuron in self.neurons:
                    neuron.basis_function(data[:self.d])
                    output += neuron.weight * neuron.y
                bias += eta * (data[self.d] - output)
                loss += (data[self.d] - output) ** 2 / 2
                for neuron in self.neurons:
                    neuron.update(eta, data[self.d] - output, data[:self.d])
            print(f'Epoch: {e + 1}/{epoch},\teta: {round(eta, 3)},\tLoss: {loss / len(self.data)}')

        print()
        print('Bias:', bias)
        print()
        print('Neurons:')
        for i in range(len(self.neurons)):
            print(f'\tNeuron {i + 1}:')
            print('\t\tweight:\t\t', self.neurons[i].weight)
            print('\t\tmean:\t\t', self.neurons[i].mean)
            print('\t\tstandard:\t', self.neurons[i].std)
        print()

        fig = plt.figure()
        plt.get_current_fig_manager().set_window_title(f'RBFN Car (train{self.d + 1}dAll.txt)')
        self.ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        self.ax.set_aspect('equal', 'box')

        print('The car is doing its best to get the finish line...')
        self.run, records, safe = [], [], True
        car, phi = [0, 0], math.pi / 2
        self.run.append([car.copy(), phi, distances(car, phi)])
        while car[1] < 34:
            dists = distances(car, phi)
            output = bias
            for neuron in self.neurons:
                if self.d == 3:
                    neuron.basis_function(dists[:, 2])
                else:
                    neuron.basis_function(np.append(car, dists[:, 2]))
                output += neuron.weight * neuron.y
            output *= -1
            rec = f'{car[0]} {car[1]} ' if self.d == 5 else ''
            rec += f'{dists[:, 2][0]} {dists[:, 2][1]} {dists[:, 2][2]} {output}\n'
            records.append(rec)
            output = math.radians(output)
            if car[0] < -3 or car[0] > 27 or car[1] < 0 or \
                    -3 < car[1] < 6 and car[0] > 3 or 6 < car[0] < 30 and car[1] < 13 or \
                    -6 < car[0] < 18 and car[1] > 19 or car[1] > 22 and car[0] < 15 or \
                    car[0] < 6 and car[1] > 10 and math.dist(car, [6, 10]) < 3 or \
                    car[0] > 18 and car[1] < 22 and math.dist(car, [6, 10]) < 3:
                safe = False
                break
            car[0] += math.cos(phi + output) + math.sin(output) * math.sin(phi)
            car[1] += math.sin(phi + output) - math.sin(output) * math.cos(phi)
            phi = (phi - math.asin(2 * math.sin(output) / 3)) % (math.pi * 2)
            self.run.append([car.copy(), phi, dists])
        with open(f'track{self.d + 1}D.txt', 'w') as file:
            file.writelines(records)
        if safe:
            print('The car arrived successfully :)')
        else:
            print('Sorry, the car accident happened... Please retry later :(')
        print('Wait a minute, result.gif is trying to come to this world...\n')
        ani = animation.FuncAnimation(fig, self.draw, frames=len(self.run) + 20)
        ani.save('result.gif', fps=10, writer='pillow')
        print('All missions are completed.')
        print(f'Please check result.gif and track{self.d + 1}D.txt :)')
        plt.show()
        os.system('pause')

    def kmeans(self, cluster) -> None:
        centers = self.data[np.random.choice(self.data.shape[0], cluster, replace=False), :self.d]
        sums = [np.zeros(self.d + 1) for _ in range(cluster)]
        diff = True
        while diff:
            diff = False
            for i in range(len(self.data)):
                for j in range(cluster):
                    dist = math.dist(self.data[i][:self.d], centers[j])
                    if dist < self.data[i][self.d + 1]:
                        diff = True
                        self.data[i][self.d + 1:self.d + 3] = dist, j
                sums[int(self.data[i][self.d + 2])][:self.d] += self.data[i][:self.d]
                sums[int(self.data[i][self.d + 2])][self.d] += 1
            for i in range(cluster):
                if sums[i][self.d] != 0:
                    centers[i] = sums[i][:self.d] / sums[i][self.d]
                    sums[i] = np.zeros(self.d + 1)
        std = [[0, 0] for _ in range(cluster)]
        for data in self.data:
            std[int(data[self.d + 2])][0] += data[self.d + 1]
            std[int(data[self.d + 2])][1] += 1
        for i in range(cluster):
            if std[i][0] != 0 and std[i][1] != 0:
                self.neurons.append(Neuron(centers[i], std[i][0] / std[i][1]))
        self.draw_3d()

    def draw_3d(self) -> None:
        if self.d != 3:
            return
        fig = px.scatter_3d(x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2], color=self.data[:, 5],
                            opacity=0.5, color_continuous_scale='jet', title='K means result')
        fig.update_traces(marker_size=3)
        fig.update_layout(scene=dict(
            xaxis_title='Front Distance',
            yaxis_title='Right Distance',
            zaxis_title='Left Distance'),
        )
        fig.show()

    def draw(self, i):
        if i < len(self.run):
            self.ax.clear()
            self.ax.set_xlim(-10, 35)
            self.ax.set_ylim(-5, 55)
            xs = [-6, -6, 18, 18, 30, 30, 6, 6, -6]
            ys = [-3, 22, 22, 50, 50, 10, 10, -3, -3]
            self.ax.plot(xs, ys, color='blue')
            self.ax.plot([-6, 6], [0, 0], color='red')
            self.ax.add_patch(Rectangle((18, 40), 12, -3, facecolor='red', fill=True))
            for run in self.run[:i]:
                self.ax.scatter(run[0][0], run[0][1], color='gold')
            self.ax.add_patch(Circle(tuple(self.run[i][0]), 3, color='red', fill=False))
            for j in range(3):
                self.ax.plot([self.run[i][0][0], self.run[i][2][j][0]],
                             [self.run[i][0][1], self.run[i][2][j][1]], color='lime')
            self.ax.text(45, 45, f'x:        {round(self.run[i][0][0], 4)}', family='Comic Sans MS', size=12)
            self.ax.text(45, 37, f'y:        {round(self.run[i][0][1], 4)}', family='Comic Sans MS', size=12)
            self.ax.text(45, 29, f'Degree: {round(math.degrees(self.run[i][1]), 4)}', family='Comic Sans MS', size=12)
            self.ax.text(45, 21, f'Front:   {round(self.run[i][2][0][2], 4)}', family='Comic Sans MS', size=12)
            self.ax.text(45, 13, f'Right:   {round(self.run[i][2][1][2], 4)}', family='Comic Sans MS', size=12)
            self.ax.text(45, 5, f'Left:     {round(self.run[i][2][2][2], 4)}', family='Comic Sans MS', size=12)


if __name__ == '__main__':
    RBFN()
