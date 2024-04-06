import pandas as pd 

df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
ID = df2['id']


drop_columns = [   
            'id',
            'bdate',
            'has_photo',
            'has_mobile',
            'education_form',
            'langs',
            'life_main',
            'people_main',
            'city',
            'last_seen',
            'occupation_type',
            'occupation_name',
            'career_start',
            'career_end',
            'education_status',
                ]

df.drop(drop_columns, axis = 1, inplace = True)

df2.drop(drop_columns, axis = 1, inplace = True)


df[list(pd.get_dummies(df['sex']).columns)] = pd.get_dummies(df['sex'])
df.drop('sex', axis = 1, inplace = True)

df2[list(pd.get_dummies(df2['sex']).columns)] = pd.get_dummies(df2['sex'])
df2.drop('sex', axis = 1, inplace = True)

#model create
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


x_train = df.drop('result', axis = 1)
y_train = df['result']
x_test = df2


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

result = pd.DataFrame({'id' : ID, 'result' : y_pred})
result.to_csv('result.csv', index = False)