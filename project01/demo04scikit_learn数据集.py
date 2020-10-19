from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

li = load_iris()

# # 获取特征值
# print("特征值：")
# print(li.data)
#
# # 获取目标值
# print("目标值：")
# print(li.target)
#
# print("描述：")
# print(li.DESCR)

# 注意返回值，既包含了训练集train x_train y_train和测试集test x_test y_test
# 注意顺序，先是特征值，后是目标值
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
print("训练集特征值和目标值：", x_train, y_train)
print("测试集特征值和目标值：", x_test, y_test)
