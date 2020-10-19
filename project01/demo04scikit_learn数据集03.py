from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

lb = load_boston()

print("特征值：")
print(lb.data)
print("目标值：")
print(lb.target)
print("描述：")
print(lb.DESCR)
