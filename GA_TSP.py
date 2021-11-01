# -*- coding: utf-8 -*-
# @Time :    2021/10/31  16:36
# @Author :  Eleven
# @Site :    
# @File :    GA_TSP.py
# @Software: PyCharm
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sko.GA import GA_TSP

# 输入点坐标，以列表形式存放
points_coordinate = [[20,16],[30,89],[49,55],[66,66],
                     [78,33],[11,90],[25,42],[22,78],
                     [12,88],[24,89],[76,13],[83,102],
                     [39,41],[81,19],[29,73],[20,10],
                     [67,66],[78,39],[24,43],[99,13],
                     [87,63],[71,91],[40,62],[77,14]]
#用np.array将格式转换，否则后面给坐标排序时会报错
points_coordinate = np.array(points_coordinate)
# 旅商要经过的地点个数
num_points = len(points_coordinate)
#将各个点的坐标以欧式距离计算后生成距离矩阵
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

#计算距离函数
def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    #个数
    num_points, = routine.shape

    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

#调用函数，迭代次数500次，prob_mut为交叉概率，size_pop为种群数量
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()
print(best_distance)    #[397.99829393]
print(best_points)  #[19 13 23 10 15  0 18  6 12  2 22 14  7  8  5  9  1 11 21  3 16 20 17 4]

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])

best_points_coordinate = points_coordinate[best_points_, :]

ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()