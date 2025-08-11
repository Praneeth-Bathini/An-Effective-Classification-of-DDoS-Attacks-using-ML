from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
# Create your views here.
def home(request):
    return render(request,'Home.html')
def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
def logout_view(request):
    logout(request)
    return redirect('login')
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
le = LabelEncoder()
global X,y,x_train,x_test,y_train,y_test
def Upload_data(request):
    load=True
    global x_train,x_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path),low_memory=False)
        df['src'] = le.fit_transform(df['src'])
        df['dst'] = le.fit_transform(df['dst'])
        df['Protocol'] = le.fit_transform(df['Protocol'])

        X = df.iloc[:,:22]
        y = df.iloc[:,-1]
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        default_storage.delete(file_path)
        outdata=df.head(100)
        return render(request,'prediction.html',{'predict':outdata.to_html()})
    return render(request,'prediction.html',{'upload':load})
Label = ['NORMAL','DDOS']
precision = []
recall = []
fscore = []
accuracy = []
import threading
def performance_metrics(algorithm, predict, testY):
    # Calculate performance metrics
    testY = testY.astype('int64')
    predict = predict.astype('int64')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Print metrics
    print(f'{algorithm} Accuracy    : {a}')
    print(f'{algorithm} Precision   : {p}')
    print(f'{algorithm} Recall      : {r}')
    print(f'{algorithm} FSCORE      : {f}')

    # Classification report
    report = classification_report(testY, predict, target_names=Label)
    print(f'\n{algorithm} classification report\n{report}')

    # Confusion matrix
    conf_matrix = confusion_matrix(testY, predict)

    # Create the plot in the main thread to avoid potential issues
    def show_plot():
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(conf_matrix, xticklabels=Label, yticklabels=Label, annot=True, cmap="Blues", fmt="g")
        ax.set_ylim([0, len(Label)])
        plt.title(f'{algorithm} Confusion matrix')
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
        plt.close()
    # Execute the plot display in the main thread
    if threading.current_thread().name == "MainThread":
        show_plot()
    else:
        # Use this if needed for threading environments
        threading.Thread(target=show_plot).start()
def LGBM_classifier(request):
    import os
    import joblib
    from lightgbm import LGBMClassifier
    global x_train,x_test,y_train,y_test
    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)

    # File path to save/load the model
    model_filename = 'model/LGBMClassifier.pkl'

    # Check if the model exists
    if os.path.exists(model_filename):
        # Load the existing model
        model = joblib.load(model_filename)
        print("LGBMClassifier model loaded successfully.")

        # Predict using the loaded model
        y_pred = model.predict(x_test)

        # Calculate and print metrics
        performance_metrics('LGBMClassifier', y_test, y_pred)
    else:
        # Train a new LGBMClassifier model
        model = LGBMClassifier(
            n_estimators=12,
            max_depth=12,
            learning_rate=9.0,
            num_leaves=2,
            min_data_in_leaf=100
        )
        model.fit(x_train, y_train)
        print("LGBMClassifier model trained successfully.")

        # Save the trained model to a file
        joblib.dump(model, model_filename)
        print("LGBMClassifier model saved successfully.")

        # Predict using the trained model
        y_pred = model.predict(x_test)

        # Calculate and print metrics
        performance_metrics('LGBMClassifier', y_test, y_pred)

    return render(request, 'prediction.html',
                  {'algorithm': 'LGBMClassifier',
                   'accuracy': accuracy[-1],
                   'precision': precision[-1],
                   'recall': recall[-1],
                   'fscore': fscore[-1]})

def RFC_classifier(request):
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # File path to save the model
    model_filename = 'random_forest_classifier_model.pkl'

    # Check if the model exists
    if os.path.exists(model_filename):
        # Load the existing model
        model = joblib.load(model_filename)
        print("Model loaded successfully.")

        # Predict using the loaded model
        y_pred = model.predict(x_test)

        # Calculate and print metrics
        performance_metrics('random_forest_classifier', y_test, y_pred)

    else:
        # Train a new Random Forest Classifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        print("Model trained successfully.")

        # Save the trained model to a file
        joblib.dump(model, model_filename)
        print("Model saved successfully.")

        # Predict using the trained model
        y_pred = model.predict(x_test)

        # Calculate and print metrics
        performance_metrics('random_forest_classifier', y_test, y_pred)
    return render(request,'prediction.html',
                  {'algorithm':'random forest cassifier',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
def prediction_view(request):
    import joblib
    Test=True
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        testdata = pd.read_csv(default_storage.path(file_path))
        test = testdata
        default_storage.delete(file_path)
        model_filename = 'random_forest_classifier_model.pkl'
        model = joblib.load(model_filename)
        test_predictions = model.predict(test)
        # Loop through each prediction and print the corresponding row
        data = []  # This will hold the rows and results
        for i, p in enumerate(test_predictions):
            print('id',i,'value:',p)
            row_data = test.iloc[i]  # Get the row data
            if p == 0:
                data.append({
                    'row': row_data, 
                    'message': f"Row {i}: ************************************************** {Label[0]}"
                })
            else:
                data.append({
                    'row': row_data, 
                    'message': f"Row {i}: **************************************************  {Label[1]}"
                })
        return render(request, 'prediction.html', {'data': data})
    return render(request,'prediction.html',{'test':Test})
