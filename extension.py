# File: my_ml_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import os
import re
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from colorama import Fore
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Example machine learning model



def get(url):
        results = []
        # Fetch HTML content
        # Parse HTML
        html_content = get_dom1(url)
        results.append(len(html_content))
        # Call tf_idf_tags and append result to list
        tf_idf_result = tf_idf_tags(html_content)
        # Call cal_similarity_score and append result to list
        similarity_score = cal_similarity_score(tf_idf_result)  # Assuming tf_idf_result is used here
        results.append(similarity_score)
        # Call extract_features and append result to list
        features = extract_features(html_content)
        # if(features['has_login_form']):
        #     features['has_login_form']=1
        # else:
        #      features['has_login_form']=0
        results.append(features['num_forms'])
        results.append(features['num_scripts'])
        results.append(features['num_hyperlinks'])
        results.append(features['num_external_hyperlinks'])
        results.append(digit_count(url))
        results.append(len(url))
        if(features['has_external_scripts']):
            features['has_external_scripts']=1
        else:
            features['has_external_scripts']=0
        results.append(features['has_external_scripts'])
        results.append(httpSecure(url))
        #results.append(features['has_login_form'])
        # results.append(features['num_js_events'])
        # feature1 = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
        # for a in feature1:
        #     features[a] = url.count(a)
        #     results.append(features[a])
        # results.append(abnormal_url(url))
        # results.append(letter_count(url))
        # results.append(Shortining_Service(url))
        # results.append(having_ip_address(url))
        return results

# Example usage
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def get_dom1(url):
    # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

    # Initialize Chrome WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(15)  # Timeout set to 10 seconds

    # Create a directory to save DOM tree file        # Open the URL with a timeout
        driver.get(url)

            # Get the DOM tree
        dom_tree = driver.page_source
        return dom_tree
       
def read_html_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

# Function to compute TF-IDF scores of HTML tags from a file
def tf_idf_tags(html_content):
        # Check if HTML content is empty
        if not html_content.strip():
            return {}  # Return an empty dictionary if content is empty

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract tags from the DOM tree
        tags = [tag.name for tag in soup.find_all()]

        # Convert tags to string for TF-IDF computation
        tag_string = ' '.join(tags)

        # Compute TF-IDF scores
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([tag_string])

        # Get feature names (tags)
        feature_names = tfidf.get_feature_names_out()

        # Create a dictionary to store TF-IDF scores for each tag
        tag_tfidf_scores = {}
        for tag, score in zip(feature_names, tfidf_matrix.toarray()[0]):
            tag_tfidf_scores[tag] = score
        return tag_tfidf_scores
def cal_similarity_score(tf_idf_tag):
        #file_path1 = "F://React_projs2//DOM_scraping_major_proj//dom_trees_new_phish//" + str(idx) + "_dom_tree.html"
        #tf_idf_tag = tf_idf_tags(file_path1)

        # Check if tf_idf_tag is empty
        if not tf_idf_tag:
            return 0  # Return 0 similarity score if tf_idf_tag is empty
    
        # Get tags from the reference document
        tags = set(res.keys()).union(set(tf_idf_tag.keys()))

        # Compute TF-IDF vectors for both documents
        vector_page1 = np.array([res.get(tag, 0) for tag in tags]).reshape(1, -1)
        vector_page2 = np.array([tf_idf_tag.get(tag, 0) for tag in tags]).reshape(1, -1)

        # Compute cosine similarity
        similarity_score = cosine_similarity(vector_page1, vector_page2)[0][0]
        return similarity_score
def extract_features(html_content):
    features = {}
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract URL features
    url = soup.find('meta', attrs={'property': 'og:url'})
    #features['url_length'] = len(url['content']) if url else 0
    
    # Extract form features
    forms = soup.find_all('form')
    features['num_forms'] = len(forms)
    #features['has_login_form'] = any(['login' in form.get('action', '').lower() for form in forms])
    
    # Extract script features # Extract hyperlink features
    scripts = soup.find_all('script')
    hyperlinks = soup.find_all('a', href=True)
    features['num_scripts'] = len(scripts)
    features['num_hyperlinks'] = len(hyperlinks)
    features['num_external_hyperlinks'] = sum(1 for link in hyperlinks if 'http' in link['href'])
    features['has_external_scripts'] = any(['http' in script.get('src', '') for script in scripts])

    js_events = soup.find_all(re.compile('^on'))
    features['num_js_events'] = len(js_events)
    
    return features
def httpSecure(url):
    htp = urlparse(url).scheme #It supports the following URL schemes: file , ftp , gopher , hdl , 
                               #http , https ... from urllib.parse
    match = str(htp)
    if match=='https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0

file_path='179_dom_tree.html'
html_content1 = read_html_file(file_path)
res=tf_idf_tags(html_content1)


with open( 'lgbm_dim_reduced.pkl', 'rb') as file:
    lgmb_loaded = pickle.load(file)

class MyModel:
     def predict(self, url_p):
        t=get(url_p)
        X = np.array(t).reshape(1, -1)
        print(X)
        y = lgmb_loaded.predict(X)
        print(y)
        prediction1 = int(y[0])
        return prediction1

app = FastAPI()
model = MyModel()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your Chrome extension's origin if known
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
class Item(BaseModel):
     url: str

# @app.options("/predict/")
# async def options_predict():
#     return {"methods": ["POST", "OPTIONS"]}

@app.post("/predict/")
async def predict(item: Item):
    prediction = model.predict(item.url)
    return {"prediction": prediction}

