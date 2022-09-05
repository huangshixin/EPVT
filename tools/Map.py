import matplotlib.pyplot as plt
import numpy as np

#绘制散点图
pvt_v2  = [70,75,80]
Epvt = [72.9,78.2,82.3]

'''
"o" 	m02 	实心圆
"v" 	m03 	下三角
"^" 	m04 	上三角
"<" 	m05 	左三角
">" 	m06 	右三角
"1" 	m07 	下三叉
"2" 	m08 	上三叉
"3" 	m09 	左三叉
"4" 	m10 	右三叉
"8" 	m11 	八角形
"s" 	m12 	正方形
"p" 	m13 	五边形
"P" 	m23 	加号（填充）
"*" 	m14 	星号
"h" 	m15 	六边形 1
"H" 	m16 	六边形 2
'''
# #设置左边名称
# plt.title('ImageNet-1k')
# plt.xlabel('ImageNet-1K Trained')
# plt.ylabel('ImageNet-1K Acc.')


x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
sizes = np.array([20,50,100,200,500,1000,60,90])
colors = np.array(["red","green","black","orange","purple","beige","cyan","magenta"])
plt.scatter(x, y, s=sizes,c=colors)
plt.show()
# plt.show()

'''
import matplotlib.pyplot as plt

x = [6, 24, 48, 72]
y1 = [87, 174, 225, 254]
y2 = [24, 97, 202, 225]
y3 = [110, 138, 177, 205]
y4 = [95, 68, 83, 105]
y5 = [72, 74, 76, 67]
plt.title('扩散速度')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('时间')  # x轴标题
plt.ylabel('差值')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3)
plt.plot(x, y3, marker='o', markersize=3)
plt.plot(x, y4, marker='o', markersize=3)
plt.plot(x, y5, marker='o', markersize=3)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y5):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['方案一', '方案二', '方案三', '方案四', '方案五'])  # 设置折线名称

'''