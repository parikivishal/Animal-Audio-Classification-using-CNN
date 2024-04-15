from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .models import user_reg
from django.http import JsonResponse
from keras.models import load_model
import numpy as np
import librosa
# Create your views here.

model = load_model("model/audio_classifier_model.h5")

def user_signup(request):
    if request.method == "POST":
        email = request.POST.get('username')
        pwd = request.POST.get('password')
        user_model = user_reg(email=email, pwd=pwd)
        try:
            user_model.save()
            return render('login')
        except:
            print("Error")
    return render(request,'signup.html')
def user_login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            # Check if user exists with the provided email and password
            user = user_reg.objects.get(email=username, pwd=password)
            # If user exists, consider authentication successful
            request.session['user_id'] = user.id  # Store user ID in session for future use if needed
            return redirect('home')  # Redirect to the home page or any other page
        except user_reg.DoesNotExist:
            # If user does not exist or password is incorrect
            error_message = "Invalid username or password. Please try again."
            return render(request, 'login.html', {'error_message': error_message})
    return render(request, 'login.html')

def user_home(request):
    return render(request,'home.html')

def extract_features(file_path, mfcc=True, n_mfcc=13, max_len=128):
    audio, _ = librosa.load(file_path, sr=22050)
    if mfcc:
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=n_mfcc)
        if (max_len > mfccs.shape[1]):
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs
    else:
        return None


def predict_audio_class(audio_file):
    # Extract features from audio file
    features = extract_features(audio_file)

    if features is None:
        return 7

    # Preprocess the features to match the input shape expected by the model
    # (You may need to reshape or normalize the features)
    # For example, if your model expects input shape (batch_size, height, width, channels):
    # features = features.reshape((1, height, width, channels))
    features = features[np.newaxis, ..., np.newaxis]

    # Make predictions
    predictions = model.predict(features)

    # Check if the prediction confidence is below a certain threshold
    threshold = 0.5  # Adjust this threshold as needed
    if np.max(predictions) < threshold:
        return 6

    # Get the predicted class

    predicted_class_index = np.argmax(predictions)

    return predicted_class_index
def user_predict(request):
    if request.method == 'POST' and request.FILES['audio']:
        audio_file = request.FILES['audio']
        predicted_class = predict_audio_class(audio_file)
        print(predicted_class)
        animals = ['Elephant', 'Leopard', 'Negative', 'Otter', 'Tiger',"Model confidence is below threshold. Unable to classify.","Unable to extract features from the audio file."]
        ans = animals[predicted_class]
        result = "Predicted Audio as "+ans
        print(result)
        return render(request, 'predict.html', {'output':result})
    return render(request,'predict.html')
