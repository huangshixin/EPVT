import matplotlib.pyplot as plt

x = [100, 200, 300]

##绘制折线图，添加数据点，设置点的大小
pvt_v2_b0 =[78.46,84.91,86.25]
epvt_b0= [85.27,89.08,90.37]
# swin_T = [84.8,90,91.1]


plt.plot(x, epvt_b0, 'r',marker='*', markersize=10)
plt.plot(x, pvt_v2_b0, 'b', marker='8',markersize=10)
# plt.plot(x,pvt_v2_b2_li,'g',marker='o', markersize=10)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('ImageNet-100 Trained')  # x轴标题
plt.xticks(x)
plt.ylabel('ImageNet-100 Acc.')  # y轴标题


#给图像添加注释，并设置样式
for a, b in zip(x, epvt_b0):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, swin_T):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, pvt_v2_b0):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

#绘制图例
# plt.legend(['EPVT', 'Swin T','PVT V2 b2'])
plt.legend(['EPVT b0','PVT V2 B0'])
#显示图像
plt.show()