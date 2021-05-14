from sklearn import tree

cls = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 10], [177, 70, 8], [160, 60, 7], [154, 54, 8], [166, 65, 9],[190, 90, 10], [175, 64, 9],
     [177, 70, 8], [159, 55, 7], [171, 75, 8], [181, 85, 10],
     [110, 40, 8], [190, 70, 9], [140, 45, 7], [130, 58, 6], [143, 80, 7],
     [164, 68, 8], [180, 58, 9]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male','female','male','female','female','female','male','male']

#training our data
cls = cls.fit(X, Y)

#Predicting 
prediction = cls.predict([[160, 90, 9]])
print(prediction)

