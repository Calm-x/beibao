import matplotlib.pyplot as plt
import numpy as np
import time

number = 100    #物品个数
capacity = 1000    #背包容量
weight = [27,21,14,82,36,94,88,71,86,69,75,60,74,16,38,49,24,68,87,85,32,
          70,37,36,65,29,91,6,54,13,78,70,95,2,38,84,30,44,85,75,39,24,42,
          76,15,53,68,78,75,24,53,84,34,99,32,22,5,80,39,92,34,59,17,41,17,
          19,28,57,31,1,21,44,47,71,28,9,15,71,36,24,36,81,33,40,69,65,68,
          73,4,19,1,11,50,13,78,93,34,60,41,35]    #各个物品的重量

profit = [75,64,68,18,83,55,60,10,18,83,53,87,80,14,92,18,67,22,64,15,80,
          33,79,81,43,93,74,48,74,72,70,94,98,57,75,98,4,46,9,80,18,96,78,
          96,24,14,37,50,52,66,63,27,30,50,88,63,100,68,50,61,55,63,47,95,
          52,40,65,43,69,34,46,26,45,18,94,93,67,3,34,79,60,67,48,65,4,9,
          28,10,49,77,98,47,56,11,23,4,70,28,95,21]    #各个物品的价值

#计算价值与重量之比
w_np = np.array(weight)
p_np = np.array(profit)
ratio = p_np / w_np
print("价值与重量之比：")
for a in ratio:
    print(format(a, '.3f'), end=" ")

#对比值进行递减排序
print("\n\n非递增排序后：")
res = sorted(ratio, reverse=True)
for b in res:
    print(format(b, '.3f'), end=" ")


#动态规划算法
class onezerobag:
    def __init__(self, w, v, c):
        self.w = w
        self.v = v
        self.c = c

    def dynamic_programming(self):
        self.v = np.array(self.v)
        self.w = np.array(self.w)
        num = self.v.size   #物体数量
        values = np.zeros([num+1, self.c+1])
        for i in range(values.shape[0]):
            values[i, 0] = 0
        for i in range(values.shape[1]):
            values[0, i] = 0
        for i in range(1, values.shape[0], 1):
            for j in range(1, values.shape[1], 1):
                if(self.w[i - 1] > j):   #如果物体重量大于包当前重量，不装进去
                    values[i,j] = values[i-1, j]
                else:
                    if(values[i-1, j] > values[i-1, j-self.w[i - 1]] + self.v[i - 1]):
                        values[i,j] = values[i-1, j]
                    else:
                        values[i,j] = values[i-1, j-self.w[i - 1]] + self.v[i - 1]
        return values

    def load_which(self, values):
        h = values.shape[0]
        c = self.c
        which = []
        for i in range(h-1, 0, -1):
            if(values[i,c] == values[i-1,c]):
                continue
            else:
                which.append(i)
                c = c - self.w[i - 1]
        which.reverse()
        return which, values[values.shape[0]-1, values.shape[1]-1]


#回溯算法
class backTrackingMethod:
    def __init__(self, w, v, c, cw, cp, bestp):
        self.w = np.array(w)
        self.v = np.array(v)
        self.c = c
        self.cw = cw
        self.cp = cp
        self.bestp = bestp

    def value_per(self):
        per = self.v / self.w
        sor = np.sort(per)
        index = np.argsort(per)

        list = []
        for i in sor:
            list.append(i)
        list.reverse()

        list1 = []
        for i in index:
            list1.append(i)
        list1.reverse()
        index = np.array(list1)
        a = self.v.copy()
        b = self.w.copy()
        for i in range(self.v.size):
            a[i] = self.v[index[i]]
            b[i] = self.w[index[i]]
        self.v = a.copy()
        self.w = b.copy()
        return self.v, self.w, index

    def bound(self, i):
        leftw = self.c - self.cw
        bestbound = self.cp
        while (i < self.v.size):
            if (self.w[i] <= leftw):
                bestbound = bestbound + self.v[i]
                leftw = leftw - self.w[i]
                i += 1
            else:
                bestbound = bestbound + self.v[i] / self.w[i] * leftw
                break
        return bestbound

    def back_tracking(self, i, visit):
        if(i > self.v.size-1):
            self.bestp = self.cp
            return

        if(self.cw + self.w[i] < self.c):
            self.cw += self.w[i]
            self.cp += self.v[i]
            visit[i] = 1
            self.back_tracking(i+1, visit)
            self.cw -= self.w[i]
            self.cp -= self.v[i]
        else:
            visit[i] = 0

        if(self.bound(i+1) >= self.bestp):
            self.back_tracking(i+1, visit)
        return visit, self.bestp

data = open("结果.txt", "w")  # 创建保存结果文件

#选择不同的算法求解0-1背包问题
print("\n\n请选择解决0-1背包问题的算法：\n1.动态规划算法\n2.回溯算法\n3.贪心算法\n")
print("请输入选项：", end=" ")
parm = input()
if parm == '1':
    start = time.time()
    question = onezerobag(weight, profit, capacity)
    x = question.load_which(question.dynamic_programming())
    end = time.time()
    print("\n***动态规划算法***")
    print("最优解序号为：")
    data.write('***动态规划算法***\n背包中所装物品为:')  # 写入文件
    for i in x[0]:
        print(i, end=" ")
        s = str(i) + " "
        data.write(s)
    a = x[1]
    print("\n最大价值为：", a)
    print("运行时间为：%.3f秒"%(end-start))
    data.write('\n最大价值为：')  # 写入文件
    data.write(str(a))
    data.write('\n运行时间为：')  # 写入文件
    data.write(str(end - start))
    data.write('秒')

elif parm == '2':
    start1 = time.time()
    visit = np.zeros(number)
    question = backTrackingMethod(weight, profit, capacity, cw=0, cp=0, bestp=0)
    weight, profit, index = question.value_per()
    visit, best = question.back_tracking(0, visit)
    end1 = time.time()
    list = []
    for i in range(visit.size):
        if (visit[i] != 0):
            list.append(index[i] + 1)
    print("\n***回溯算法***")
    print("最优解序号为：")
    data.write('***回溯算法***\n背包中所装物品为：')  # 写入文件
    for a in sorted(list):
        print(a, end=" ")
        s = str(a) + " "
        data.write(s)
    print("\n最大价值为：", best)
    print("运行时间为：%.3f秒"%(end1-start1))
    data.write('\n最大价值为：')  # 写入文件
    data.write(str(best))
    data.write('\n运行时间为：')  # 写入文件
    data.write(str(end1 - start1))
    data.write('秒')

else:
    print("出错！没有这个选项！")


plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(weight, profit, s=20)
plt.xlabel("Weight", fontsize=12, color="r")
plt.ylabel("Profit", fontsize=12, color='r')
plt.show()