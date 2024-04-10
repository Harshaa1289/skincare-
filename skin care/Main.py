from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from flask import Flask

app = Flask(__name__)


app.secret_key = 'welcome'
global user, rf, image_data
accuracy = []
precision = []
recall = []
fscore = []

def getText(img_name):
    data =  pytesseract.image_to_string(img_name, lang='eng')
    data = data.replace("\n"," ")
    return data

dataset = pd.read_csv(r'C:\Users\Harsha\OneDrive\Desktop\SkinCare\Dataset\cosmetics5.csv')
ingredients = dataset['Ingredients'].ravel()
dataset.drop(['Ingredients', 'Brand', 'Name', 'Price', 'Rank'], axis = 1,inplace=True)
labels = np.unique(dataset['Label'])
le = LabelEncoder()
dataset['Label'] = pd.Series(le.fit_transform(dataset['Label'].astype(str)))#encode all str columns to numeric
dataset.fillna(0, inplace = True)
Y = dataset['Label'].ravel()

dataset.drop(['Label'], axis = 1,inplace=True)
dataset = dataset.values
print(dataset.shape)
tfidf_vectorizer = TfidfVectorizer(lowercase=True, token_pattern='[a-zA-Z]+', max_features=800)
X = tfidf_vectorizer.fit_transform(ingredients).toarray()
#df = pd.DataFrame(X, columns=tfidf_vectorizer.get_feature_names_out())
#print(df)

X = np.hstack((X, dataset))
print(X.shape)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

def calculateMetrics(algorithm, predict, y_test):
    label = ['Normal Event', 'Disaster Event']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

svm_cls = svm.SVC() #create SVM object
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)
    
dt_cls = DecisionTreeClassifier() #create Decision Tree object
dt_cls.fit(X_train, y_train)
predict = dt_cls.predict(X_test)
calculateMetrics("Decision Tree", predict, y_test)

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global rf, image_data, tfidf_vectorizer, labels
        combination = float(request.form['t1'].strip())
        dry = float(request.form['t2'].strip())
        normal = float(request.form['t3'].strip())
        oily = float(request.form['t4'].strip())
        sensitive = float(request.form['t5'].strip())
        acne = float(request.form['t6'].strip())
        pimples = float(request.form['t7'].strip())
        black_head = float(request.form['t8'].strip())
        dull = float(request.form['t9'].strip())
        wrinkles = float(request.form['t10'].strip())
        dark = float(request.form['t11'].strip())
        pollution = float(request.form['t12'].strip())
        blue = float(request.form['t13'].strip())
        uv = float(request.form['t14'].strip())

        features = []
        features.append([combination, dry, normal, oily, sensitive, acne, pimples, black_head, dull, wrinkles, dark, pollution, blue, uv])
        features = np.asarray(features)
        print(features.shape)
        vector = tfidf_vectorizer.transform([image_data]).toarray()
        print(vector.shape)
        vector = np.hstack((vector, features))
        print(vector.shape)

        predict = rf.predict(vector)[0]
        predict = int(predict)
        print(predict)
        predict = labels[predict] 
        return render_template('UserScreen.html', data="Product Suitability = "+predict)
        
    
@app.route('/UploadImageAction', methods=['GET', 'POST'])
def UploadImageAction():
    if request.method == 'POST':
        global image_data
        file = request.files['t1']
        filename = secure_filename(file.filename)
        print(filename)
        if os.path.exists("static/"+filename):
            os.remove("static/"+filename)
        file.save("static/"+filename)
        image_data = getText("static/"+filename)
        return render_template('UserInput.html', data="Choose your Skin Type")


@app.route('/UploadImage', methods=['GET', 'POST'])
def UploadImage():
    return render_template('UploadImage.html', msg='')

@app.route('/TrainML', methods=['GET', 'POST'])
def TrainML():
    global user, accuracy, precision, recall, fscore
    command='<table border=1 align=center>'
    command+='<tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th></tr>'
    color = '<font size="" color="black">'
    algorithms = ['Random Forest', 'SVM', 'Decision Tree']
    for i in range(len(algorithms)):
        command+='<tr><td>'+color+algorithms[i]+'</td><td>'+color+str(accuracy[i])+'</td><td>'+color+str(precision[i])+'</td><td>'+color+str(recall[i])+'</td><td>'+color+str(fscore[i])+'</td></tr>'
    command += "</table><br/><br/><br/>"
    return render_template('UserScreen.html', data=command)

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/Login', methods=['GET', 'POST'])
def Login():
   return render_template('Login.html', msg='')

@app.route('/LoginAction', methods=['GET', 'POST'])
def LoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', data="Welcome "+user)
        else:
            return render_template('Login.html', data="Invalid login details")

@app.route('/')
def root():
    return redirect(url_for('index')) 



if __name__ == "__main__":

    app.run(debug=True)