import matplotlib.pyplot as plt

x = [6, 24, 48, 72]
y1 = [87, 174, 225, 254]
y2 = [24, 97, 202, 225]
y3 = [110, 138, 177, 205]
y4 = [95, 68, 83, 105]
y5 = [72, 74, 76, 67]
plt.title('扩散速度')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('ImageNet-1K Trained')  # x轴标题
plt.ylabel('ImageNet-1K Acc.')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3)
plt.plot(x, y3, marker='o', markersize=3)
plt.plot(x, y4, marker='o', markersize=3)
plt.plot(x, y5, marker='o', markersize=3)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=5)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y5):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['PVT v2', 'Swin Transformer', 'ResNets', 'EPVT', 'VIT'])  # 设置折线名称

plt.show()  # 显示折线图
