import streamlit as st
import pandas as pd
from pycaret.classification import *
import base64

# Load model
model = load_model('my_best_pipeline_cluster_kcn4')

# Create function for prediction
def predict(df):
    preds = predict_model(model, data=df)['Label']
    scores = predict_model(model, data=df)['Score']
    df['Label'] = preds
    df['Score'] = scores
    return df

# Define Streamlit app
def main():
    st.title('Nhận diện KCN, KCX từ tọa độ input')

    # Allow user to input longitude and latitude
    longitude = st.text_input('Longitude')
    latitude = st.text_input('Latitude')

    # If user inputs longitude and latitude, make predictions and show results
    if st.button('Predict'):
        if longitude and latitude:
            df = pd.DataFrame({'longitude': [float(longitude)], 'latitude': [float(latitude)]})
            predictions = predict(df)
            st.write(predictions)
        else:
            st.warning('Please input both longitude and latitude.')

    # Allow user to upload csv file
    uploaded_file = st.file_uploader('Or upload a CSV file:', type=['csv'])

    # If user uploaded a file, make predictions and show results
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predictions = predict(df)
        st.write(predictions)

        # Add download button for csv file
        csv = predictions.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download csv file</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
