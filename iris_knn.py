from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def knn_iris():
    
    # 1）get the iris data
    iris = load_iris()

    # 2）split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）standard
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）KNN Estimator
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5）Model Evaluation
    # Method One：Comparing the estimated value with the true value
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print(y_test == y_predict)

    # Method Two：Accuracy
    score = estimator.score(x_test, y_test)
    print("The accuracy of KNN is：\n", score)

    return None


def knn_iris_gscv():
    """
    KNN Classifier，using Cross_Validation and Grid_searching methods
    :return:
    """
    # 1）ge the iris data
    iris = load_iris()

    # 2）split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）standard
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）KNN Estimator
    estimator = KNeighborsClassifier()

    # Cross_Validation and Grid_searching
    # parameters
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5）Model Evaluation
    # Method one
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("comapring the estimated value with the true values:\n", y_test == y_predict)

    # Method two
    score = estimator.score(x_test, y_test)
    print("the Accuracy is：\n", score)

    # 最佳参数：best_params_
    print("optimal parameter：\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("optimal result：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("optimal estimator:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("CV results:\n", estimator.cv_results_)

    return None



if __name__ == "__main__":
    # 代码1： 用KNN算法对鸢尾花进行分类
    # knn_iris()
    # 代码2：用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    # knn_iris_gscv()
    # 代码3：用朴素贝叶斯算法对新闻进行分类
    # nb_news()
    # 代码4：用决策树对鸢尾花进行分类
    knn_iris()
    knn_iris_gscv()
