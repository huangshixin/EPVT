import matplotlib.pyplot as plt

plt.title('Classfication Accuracy of EPVT-T with different filter sizes')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Epoch')  # x轴标题
plt.ylabel('ImageNet-100 Acc.')  # y轴标题

x = [50, 100,150,200, 250,300]
y1 = [78.25, 85.27, 87.66, 88.95,89.81,90.25]#with 3x3
y2 = [78.01, 84.27, 86.84, 88.17,88.35,89.27]
y3 = [76.10, 83.67, 86.03, 87.58,88.46,88.76]
y4 = [77.15, 84.17, 87.35, 88.87,89.67,89.84]#池化3x3

plt.plot(x, y1, marker='o', markersize=4)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='P', markersize=4)
plt.plot(x, y3, marker='*', markersize=4)
plt.plot(x, y4, marker='H', markersize=4)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=5)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['with 3x3 filter', 'with 5x5 filter', 'with 7x7 filter', 'with 3x3 pooling'])  # 设置折线名称

plt.show()  # 显示折线图