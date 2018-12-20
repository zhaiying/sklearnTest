from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.metrics


loaded_data = datasets.load_boston()

data_X = loaded_data.data
data_y = loaded_data.target

import pandas as pd
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

#实例化1个DataFrame对象赋值给变量df，DataFrame对象类似于Excel表格。

df = pd.DataFrame(data_X, columns=loaded_data.feature_names)
#查看变量df的前10行
df.head(10)
#df中是否有空值，如果有空值，则需要对其进行处理
df.info()
#查看变量df中各个字段的计数、平均值、标准差、最小值、下四分位数、中位数、上四分位、最大值
df.describe().T

#print(df.describe().T)

#使用matplotlib库画图时，导入画板对象plt和防止中文出现乱码，一定要先运行下面3行代码
import matplotlib.pyplot as plt

myfont = plt.font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.scatter(df['CRIM'], data_y)
plt.title('城镇人均犯罪率与房价散点图')
plt.xlabel('城镇人均犯罪率')
plt.ylabel('房价')
plt.show()








# X_train,X_test,y_train,y_test=train_test_split(data_X,data_y,test_size=0.3)
#
# # print(y_train)
#
# model=LinearRegression()
# model.fit(X_train, y_train)
#
# X=model.predict(X_test)
#
# print(X,y_test)
# acc_score = sklearn.metrics.accuracy_score(y_test,X)
# print(acc_score)

# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)
# plt.scatter(X, y)
# plt.show()