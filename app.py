import streamlit as st
import plotly.express as px
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Iris Pridictor",
                   layout="centered")
st.title("Iris Flower Prediction Model")
st.write("This app predicts the Iris flower species based on user-provided measurements using the Random Forest Classification algorithm.")
df=pd.read_csv('iris.csv')

img1 = Image.open("mlproject/iris_setosa.png").resize((250, 200))
img2 = Image.open("mlproject/iris_versicolor.png").resize((250, 200))
img3 = Image.open("mlproject/iris_virginica.png").resize((250, 200))


c1, c2, c3 = st.columns(3)
with c1:
     st.image(img1, caption="Iris Setosa")
with c2:
    st.image(img2, caption="Iris Versicolor")
with c3:
    st.image(img3, caption="Iris Virginica")
with st.expander("Click to Check DataSet"):
    df
st.sidebar.title("User Input Parameters")
sepal_length = st.sidebar.slider("Sepal Length in cm",
               float( df["sepal_length_cm"].min()),
               float (df["sepal_length_cm"].max()),
               float(df["sepal_length_cm"].mean()),
               )
sepal_width = st.sidebar.slider("Sepal Width in cm",
              float (df["sepal_width_cm"].min()),
              float (df["sepal_width_cm"].max()),
              float (df["sepal_width_cm"].mean()),
              )
petal_length = st.sidebar.slider("Petal Length in cm",
                float(df["petal_length_cm"].min()),
                float(df["petal_length_cm"].max()),
                float(df["petal_length_cm"].mean()),
              )

petal_width = st.sidebar.slider("Petal Width in cm",
                float(df["petal_width_cm"].min()),
                float(df["petal_width_cm"].max()),
                float(df["petal_width_cm"].mean()),
              )
features = ["sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm"]
X = df[features]
y = df["species"]
input_data = pd.DataFrame(
                [[sepal_length,sepal_width,petal_length,petal_width]],
                columns = features
                )
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    # Stratify means we are dividing data proportionally in both training and test 
    # with stratify = y each specie will be represented in both training and testing 
    #proposionally 80%,20%
    stratify=y

)
model = RandomForestClassifier(n_estimators=200, random_state=42) #n_estimator means number of decision trees
model.fit(X_train, y_train)
test_prediction = model.predict(X_test)
accuracy = accuracy_score(y_test,test_prediction)
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
probability_table = pd.DataFrame(
                       {"species": model.classes_,"probability": probabilities}).set_index('species')
feature_importance = pd.DataFrame(
                                {
                                    'Features' : features,
                                    'Importance' : model.feature_importances_
                                }
                                    )

confidence = probabilities.max()

col1,col2,col3 = st.sidebar.columns(3)
with col2:
    predict_button = st.button("Predict")
if predict_button:
    st.success(f"{prediction.title()}")
    with c1:
            st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1%}"
                     )
    with c2: 
            st.metric(
                    "Training Sample",
                    len(X_train),
                    delta=len(X_train)+len(y_train)
            )
    with c3:
        st.metric(label = 'Accuray Score', value=(f"{accuracy:.1%}"))
    
    st.subheader("Probabilites:")
    st.bar_chart(probability_table)
   
    #st.bar_chart(feature_importance,x="Features",y='Importance',orientation='h')
    fig = px.bar(
                feature_importance,
                y="Features",
                x="Importance",
                orientation = 'h',
                color_discrete_sequence=["#6C63FF"]
                )
    st.plotly_chart(fig)
    
    


