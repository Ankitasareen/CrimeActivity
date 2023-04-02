from cv2 import dft
from sklearn.metrics import label_ranking_average_precision_score
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import streamlit as st
import numpy as np
import tempfile
import cv2
import time
from bokeh.models.widgets import Div
from keras.models import load_model
import pickle
import requests
from PIL import Image
from keras.models import load_model
from collections import deque
import numpy as np
import ast
import pickle
import sklearn
import cv2
import os 
from sqlconnector import *
global label
from sqlalchemy import create_engine

import pymysql
import streamlit.components.v1 as stc

DEMO_VIDEO = 'shooting.mp4'
st.title('Criminal Detection')


model = load_model("ClassModel")
lb = pickle.loads(open("ClassModelbinarizer.pickle", "rb").read())

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Crime Detection')
st.sidebar.subheader(
    'ABCDEFGHIJKL')


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Crime Detector']
                                )

if "visibility" not in st.session_state:
       st.session_state.visibility = "visible"
       st.session_state.disabled = False

text_input = st.text_input(
        "Enter the location 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,)
print(text_input)


if app_mode == 'About App':

   
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**How does the Crminal detector work?**")
    st.markdown(''' Model ''')


elif app_mode == 'Crime Detector':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    #nav search bar
    
    sqlEngine       = create_engine('mysql+pymysql://root:1234@localhost/Criminal_Activity_Detection')
    dbConnection    = sqlEngine.connect()
    #results

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    

    #writer = None
    while cap.isOpened():
        (taken, frame) = cap.read()
        if not taken:
            break

        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
        Q = deque(maxlen=256)

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        #print(lb.classes_)
        #print(label)
        #print(type(label[0]))
        text = "Detected: {}".format(label)
        #st.markdown(text)
        #myset.add(label)
        cv2.imshow('in progress', output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if  label == 'weapon' or label =='normds':
          break

        if record:
            #st.checkbox("Recording", value=True)
            out.write(frame)
        

        stframe.image(frame, clamp=True, use_column_width=True)
    # st.markdown(text)
    st.text('Video Processed')
    #st.markdown(text)
    cap.release()
    out.release()
    
    
    
    #creating table
    create_shoplifting_table = """
     CREATE TABLE test (
        ID INT AUTO_INCREMENT PRIMARY KEY ,
        PREDICTION varchar(255), 
        CREATION_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        LOCATION varchar(255)

    );
   """

    pw="1234"
    db="Criminal_Activity_Detection"
    connection = create_db_connection("localhost", "root", pw, db) # Connect to the Database
    execute_query(connection, create_shoplifting_table) # Execute our defined query
    
    ##populating the table
    pop_shoplifting = f"""
                         INSERT INTO test (PREDICTION, LOCATION) 
                         VALUES ("{label}", "{text_input}");
                      """
    

    connection = create_db_connection("localhost", "root", pw, db)
    execute_query(connection, pop_shoplifting)
    print(pop_shoplifting)
    data= pd.read_sql('SELECT * FROM test', dbConnection)
    df = pd.DataFrame(data)
    st.dataframe(df)

    with st.form(key='searchform'):
         nav1,nav2,nav3 = st.columns([3,2,1])

         with nav1:
              search_term = st.text_input("Crime")
         with nav2:
              location = st.text_input("Location")
         with nav3:
              st.text("Search ")
              submit_search = st.form_submit_button(label='Search')

    st.success("You searched for {} in {}".format(search_term,location))

    if submit_search:
        query = f"""
                    SELECT *
                    FROM test
                    WHERE LOCATION="{location}" 
                    AND PREDICTION = "{search_term}";
                """
        getdata=pd.read_sql(query, dbConnection)

        #no. of results
        num_of_results=len(getdata)
        st.text("Showing {} results..".format(num_of_results))

        df = pd.DataFrame(getdata)
        #print(df)
        st.dataframe(df)
        
    connection.close() 
    

    #print(myset)
    #st.markdown(text)
    # cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
    # (255, 0, 0), 5)