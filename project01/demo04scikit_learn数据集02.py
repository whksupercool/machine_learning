from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups(subset='all')

print(news.data)
print(news.target)
