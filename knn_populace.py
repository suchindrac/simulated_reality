import math
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys

def high():
    return random.randint(7, 10)

def low():
    return random.randint(2, 5)

if __name__ == "__main__":
    points_df = pd.DataFrame({
        'User':       [10, 3, 2, 4],
        'Greedy':     [1, 9, 2, 4],
        'Sadist':     [3, 2, 7, 2],
        'Careless':   [2, 3, 3, 8],
        'Feeling':    [3, 2, 2, 2],
        'Type':       ["User", "Greedy", "Sadist", "Careless"]
    })

    extend = True
    if extend:
        columns = ["User", "Greedy", "Sadist", "Careless"]
        
        i = 0
        while i < 97:
            values = []
            for column in columns:
                if column == "User":
                    df = pd.DataFrame({"User": [high()], "Greedy": [low()],
                               "Sadist": [low()], "Careless": [low()],
                               "Feeling": [low()], "Type": ["User"]})
                elif column == "Greedy":
                    df = pd.DataFrame({"User": [low()], "Greedy": [high()],
                               "Sadist": [low()], "Careless": [low()],
                               "Feeling": [low()], "Type": ["Greedy"]})
                elif column == "Sadist":
                    df = pd.DataFrame({"User": [low()], "Greedy": [low()],
                               "Sadist": [high()], "Careless": [low()],
                               "Feeling": [low()], "Type": ["Sadist"]})
                elif column == "Careless":
                    df = pd.DataFrame({"User": [low()], "Greedy": [low()],
                               "Sadist": [low()], "Careless": [high()],
                                "Feeling": [low()], "Type": ["Careless"]})
                i += 1
                
                points_df = pd.concat([points_df, df], ignore_index=True)

        i = 0
        while i < 500:
            values = []
            for column in columns:
                if column == "User":
                    df = pd.DataFrame({"User": [high()], "Greedy": [low()],
                               "Sadist": [low()], "Careless": [low()],
                               "Feeling": [high()], "Type": ["User"]})
                elif column == "Greedy":
                    df = pd.DataFrame({"User": [low()], "Greedy": [high()],
                               "Sadist": [low()], "Careless": [low()],
                               "Feeling": [high()], "Type": ["Greedy"]})
                elif column == "Sadist":
                    df = pd.DataFrame({"User": [low()], "Greedy": [low()],
                               "Sadist": [high()], "Careless": [low()],
                               "Feeling": [high()], "Type": ["Sadist"]})
                elif column == "Careless":
                    df = pd.DataFrame({"User": [low()], "Greedy": [low()],
                               "Sadist": [low()], "Careless": [high()],
                                "Feeling": [high()], "Type": ["Careless"]})
                i += 1
                
                points_df = pd.concat([points_df, df], ignore_index=True)

#             
#     x = points_df.loc[:,"User"]
#     y = points_df.loc[:,"Greedy"]
#     z = points_df.loc[:,"Sadist"]
#     c = points_df.loc[:,"Careless"]
# 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
#     fig.colorbar(img)
#     
#     plt.show()
#     sys.exit(1)
    
    input_cols = ['User', 'Greedy', 'Sadist', 'Careless']
    target_cols = "Feeling"

    train_inputs = points_df[input_cols].copy()
    train_targets = points_df[target_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(train_inputs, train_targets,
                                                         random_state=104,test_size=0.25, shuffle=True)

    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)

    result = []
    for e in test_probs:
        if np.argmax(e) >= 7:
            result.append("Feeling Good")
        else:
            result.append("Feeling Bad")

    print(max(result))
    
