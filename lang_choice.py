import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

data = pd.read_csv('lang_train.csv', encoding = "shift-jis")
data2 = pd.read_csv('lang_test.csv', encoding = "shift-jis")
x_train = data.drop(['rank','year'], axis = 1)
y_train = data['rank']
x_test = data2.drop(['rank','year'], axis = 1)
y_test = data2['rank']


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, random_state = 10, max_depth = 4)
forest.fit(x_train, y_train)

print(list(data2['rank']))
print(forest.predict(x_test))

importance = pd.DataFrame({"variables":x_train.columns, "importance":forest.feature_importances_})
print(importance)

from sklearn import metrics
print(metrics.accuracy_score(y_train, forest.predict(x_train)))
print(metrics.accuracy_score(y_test, forest.predict(x_test)))

def plot_feature_importance_x_train(model):
    n_features = x_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, color = "blue")
    plt.yticks(np.arrange(n_features), x_train.columns, fontsize = 15)
    plt.xlabel("feature_importance", fontsize = 25)
    plt.ylabel("feature", fontsize = 25)

plt.figure(figsize = (15,7))
plot_feature_importance_x_train(forest)
