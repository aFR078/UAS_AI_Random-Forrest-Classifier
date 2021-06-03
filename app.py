import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import shap
import matplotlib.pyplot as plt


image = Image.open('Robot-Unicorn.jpg')

st.write("""
# Startup Investment App
This app predicts the **success of startups**!\n
Data acquired from Kaggle originally sourced from [Crunchbase](https://www.kaggle.com/justinas/startup-investments).
""") 


st.image(image, use_column_width=True)

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        startup_age = st.sidebar.slider('Startup age (years)', 0.2,10.0,5.0)
        twitter_account = st.sidebar.selectbox('Twitter account',('Yes','No'))
        category_code = st.sidebar.selectbox('category', ('software', 'web','advertising','cleantech','games_video','mobile','health',
                                            'analytics','network_hosting','consulting','finance','medical','manufacturing',
                                            'enterprise','other','ecommerce','public_relations','hardware','education','news','government',
                                            'security','biotech','photo_video','search','travel','semiconductor','social','legal',
                                            'transportation','hospitality','nonprofit','sports','fashion','messaging','music','automotive',
                                            'design','real_estate','local','nanotech','pets'))
        first_funding_at = st.sidebar.slider('First funding received at (months)', 0.0,120.0,15.0)
        last_funding_at = st.sidebar.slider('Last funding received at (months)', 0.0,180.0,21.0)
        funding_rounds = st.sidebar.slider('Total funding rounds', 0,10,1)
        funding_total_usd = st.sidebar.slider('Total funding raised (usd in millions)', 0,500,50)
        first_milestone_at = st.sidebar.slider('First milestone at (months)', 0.0,120.0,23.0)
        last_milestone_at = st.sidebar.slider('Last milestone at (months)', 0.0,180.0,27.0)
        milestones = st.sidebar.slider('Total number of milestones', 0,9,2)
        relationships = st.sidebar.slider('Number of relationships', 0,1000,4)
        
        graduated_at =st.sidebar.slider('difference between startup inception date and founder graduation date (in months)', -564.0,684.0,36.0)
        angel=st.sidebar.slider('Amount raised through angels (usd in millions)', 0.0,35.0,0.025)
        seed=st.sidebar.slider('Amount raised at seed stage (usd in millions)', 0.0,200.0,0.1)
        grant=st.sidebar.slider('Amount raised using grants (usd in millions)', 0.0,0.1,0.01)
        crowd_equity=st.sidebar.slider('Amount raised through crowd funding (usd in millions)', 0.0,5.0,0.0)
        series_a=st.sidebar.slider('Amount raised at series a (usd in millions)', 0.0,100.0,1.0)
        series_b=st.sidebar.slider('Amount raised at series b (usd in millions)', 0.0,500.0,10.0)
        series_c=st.sidebar.slider('Amount raised at series c (usd in millions)', 0.0,5000.0,50.0)

        data = {'category_code': category_code,
                'first_funding_at': first_funding_at,
                'last_funding_at': last_funding_at,
                'funding_rounds': funding_rounds,
                'funding_total_usd': funding_total_usd,
                'first_milestone_at': first_milestone_at,
                'last_milestone_at':last_milestone_at,
                'milestones': milestones,
                'relationships': relationships,
                'graduated_at': graduated_at,
                'angel': angel,
                'seed': seed,
                'grant': grant,
                'crowd_equity': crowd_equity,
                'series_a': series_a,
                'series_b': series_b,
                'series_c': series_c,
                'twitter_account': twitter_account,
                'startup_age': startup_age}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()




input_df['category_code'] = np.where(input_df.category_code == 'web', 1, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'advertising', 2, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'cleantech', 3, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'games_video', 4, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'mobile', 5, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'health', 6, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'software', 7, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'analytics', 8, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'network_hosting', 9, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'consulting', 10, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'finance', 11, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'medical', 12, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'manufacturing', 13, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'enterprise', 14, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'other', 15, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'ecommerce', 16, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'public_relations', 17, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'hardware', 18, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'education', 19, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'news', 20, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'government', 21, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'security', 22, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'biotech', 23, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'photo_video', 24, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'search', 25, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'travel', 26, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'semiconductor', 27, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'social', 28, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'legal', 29, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'transportation', 30, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'hospitality', 31, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'nonprofit', 32, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'sports', 33, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'fashion', 34, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'messaging', 35, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'music', 36, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'automotive', 37, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'design', 38, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'real_estate', 39, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'local', 40, input_df.category_code)
input_df['category_code'] = np.where(input_df.category_code == 'nanotech', 41, input_df.category_code)

input_df['twitter_account'] = np.where(input_df.twitter_account == 'Yes', 1, input_df.twitter_account)
input_df['twitter_account'] = np.where(input_df.twitter_account == 'No', 0, input_df.twitter_account)



input_df['funding_total_usd'] = input_df['funding_total_usd']*1000000
input_df['angel'] = input_df['angel']*1000000
input_df['seed'] = input_df['seed']*1000000
input_df['grant'] = input_df['grant']*1000000
input_df['crowd_equity'] = input_df['crowd_equity']*1000000
input_df['series_a'] = input_df['series_a']*1000000
input_df['series_b'] = input_df['series_b']*1000000
input_df['series_c'] = input_df['series_c']*1000000


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(input_df)


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
startup_base = pd.read_csv('Streamlit_data.csv')
startupp = startup_base.drop(columns=[  'closed_at', 'country_code', 'city','series_d',
       'series_e', 'series_f', 'series_g', 'debt_round', 'private_equity',
       'convertible', 'crowd', 'post_ipo_equity', 'secondary_market',
       'post_ipo_debt', 'unattributed', 'twitter_account', 'satus_cat'])

df = pd.concat([input_df,startupp],axis=0) 

data = df.drop(columns=['labels'])


#Filling null values
for featuree in ['category_code', 
       'first_funding_at', 'last_funding_at', 'funding_rounds',
       'funding_total_usd', 'first_milestone_at', 'last_milestone_at',
       'milestones', 'relationships', 'graduated_at', 'angel', 'seed', 'grant',
       'crowd_equity', 'series_a', 'series_b', 'series_c','twitter_account',
       'startup_age']:
    data[featuree] = data[featuree].fillna(0)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=30)
km.fit(data)

cluster_labels = km.labels_
df['cluster_labels']=cluster_labels
df = df.drop(columns=['labels'])


df1 = df[:1] # Selects only the first row (the user input data)




# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df1)
else:
    st.write('Please upload cvs file or adjust parameters as required.')
    st.write(df1)

# Reads in saved classification model
load_clf = pickle.load(open('RFK_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df1)
prediction_proba = load_clf.predict_proba(df1)


st.subheader('Prediction')
labels = np.array(['Unsuccessful','Successful'])
st.write(labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba *100)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(load_clf)
shap_values = explainer.shap_values(df1)

st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, df1) 
# st.pyplot(bbox_inches='tight')
# st.write('---')

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, df1, plot_type="bar")
st.pyplot(bbox_inches='tight')



