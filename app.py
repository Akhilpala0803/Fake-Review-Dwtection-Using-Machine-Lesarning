import pickle

import numpy as np
import pandas as pd
import sklearn as sk
from flask import Flask, jsonify, render_template, request
from sklearn import svm 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the SVM model (you need to train and save the SVM model first)
model = pickle.load(open('svm_model.pkl', 'rb'))


@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df = pd.read_csv('deceptive-opinion.csv')
    df1 = df[['deceptive', 'text']]
    df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
    df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
    X = df1['text']
    Y = np.asarray(df1['deceptive'], dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)

    # Train the SVM model has performed.
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y_train)

    message = request.form.get('enteredinfo')
    data = [message]
    vect = cv.transform(data).toarray()
    pred = clf.predict(vect)

    return render_template('result.html', prediction_text=pred)

if __name__ == "_main_":
    app.run(debug=True)
    import random
import matplotlib.pyplot as plt

# A simple function to determine if a review is fake or real based on keywords
def is_fake_review(review):
    fake_keywords = ["free", "giveaway", "discount", "sponsored", "ad"]
    # Check if any fake keyword is in the review (returns True if fake, otherwise False)
    return any(keyword in review.lower() for keyword in fake_keywords)

# Collect reviews from users
def collect_reviews():
    reviews = []
    num_reviews = int(input("Enter the number of reviews you want to check wheather it's real or fake: "))
    
    for i in range(num_reviews):
        review = input(f"Enter review {i + 1}: ")
        reviews.append(review)
    
    return reviews

# Process reviews and determine fake/real counts
def process_reviews(reviews):
    real_count = 0
    fake_count = 0
    
    for review in reviews:
        if is_fake_review(review):
            fake_count += 1
        else:
            real_count += 1
    
    return real_count, fake_count

# Visualize the results with a pie chart
def show_graph(real_count, fake_count):
    labels = 'Real Reviews', 'Fake Reviews'
    sizes = [real_count, fake_count]
    colors = ['#66b3ff', '#ff6666']
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Real vs Fake Reviews")
    plt.show()

# Main function to simulate the fake review system
def fake_review_system():
    # Step 1: Take product link as input
    product_link = input("Enter the product link: ")
    print(f"Collecting reviews for product: {product_link}")
    
    # Step 2: Collect reviews
    reviews = collect_reviews()
    
    # Step 3: Process reviews to detect real/fake
    real_count, fake_count = process_reviews(reviews)
    
    # Step 4: Display results
    print(f"Number of Real Reviews: {real_count}")
    print(f"Number of Fake Reviews: {fake_count}")
    
    # Step 5: Show graph
    show_graph(real_count, fake_count)

# Run the system
fake_review_system()



# app.py
from flask import Flask, request, render_template, redirect, url_for
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the link from the form
    link = request.form['link']
    
    # Fake logic to classify reviews (replace this with your real logic)
    # Assuming we have 100 reviews for simplicity
    total_reviews = 100
    fake_reviews = random.randint(20, 80)  # Random number of fake reviews
    real_reviews = total_reviews - fake_reviews
    
    # Calculate percentages
    fake_percentage = (fake_reviews / total_reviews) * 100
    real_percentage = (real_reviews / total_reviews) * 100

    # Generate a pie chart
    labels = ['Fake Reviews', 'Real Reviews']
    sizes = [fake_percentage, real_percentage]
    colors = ['#ff6666', '#66b3ff']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Save the plot to a string buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', plot_url=plot_url, link=link)

if __name__ == '__main__':
    app.run(debug=True)
