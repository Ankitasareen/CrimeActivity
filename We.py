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
import streamlit_authenticator as stauth
from pathlib import Path
from sqlconnector import *
global label
from sqlalchemy import create_engine
import pymysql
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

#user autentication
names=["khushi","ankita"]
usernames=["k3107","ank11"]
file_path = Path("_file_").parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,"crime_activity","abcdef",cookie_expiry_days=30)
name,authentication_status,username = authenticator.login("login","main")



if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    
    
    DEMO_VIDEO = 'shooting.mp4'
    st.title('Criminal Activity Detection')
    sqlEngine       = create_engine('mysql+pymysql://root:1234@localhost/Criminal_Activity_Detection')
    dbConnection    = sqlEngine.connect()


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
    

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}!")
    st.sidebar.title('Choose the App mode')  
 


    app_mode = st.sidebar.selectbox( '',['About App',"Live Feed", 'Crime Detector','Search'])
    if app_mode == 'About App':

    
        st.markdown(
            """
        <style>
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
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

        st.markdown("**This app helps in identifying criminal activity in the video footage and stores all the information in the database which can be further studied.**")
    
    elif app_mode == "Live Feed":
        st.header("Webcam Live Feed")
        st.write("Click on start to use camera for live monitoring!")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)

    elif app_mode == 'Crime Detector':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        text_input = st.text_input(
            "Enter the location 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,)
        
        text_input1 = st.text_input(
            "Enter the Camera ID",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,)
        
        use_webcam = st.sidebar.button('Use Camera')
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
        if video_file_buffer is not None:
            file_name = video_file_buffer.name
            st.write("Video file name:", file_name)
        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            if use_webcam:
                cap = cv2.VideoCapture(0)
                print("aaaa")
            else:
                cap = cv2.VideoCapture(DEMO_VIDEO)
                tfflie.name = DEMO_VIDEO

        else:
            tfflie.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tfflie.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
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
            

            st.image(frame, clamp=True, use_column_width=True)
            st.write("webcam")
        # st.markdown(text)
        st.text('Video Processed')
        
        with open("output.mp4", "rb") as file:
            btn = st.download_button(
            label="Download video",
            data=file,
            file_name="output.mp4",
            mime="video/mp4"
          )
        #st.markdown(text)
        cap.release()
        out.release()
        
        sqlEngine       = create_engine('mysql+pymysql://root:1234@localhost/Criminal_Activity_Detection')
        dbConnection    = sqlEngine.connect()
        #creating table
        create_shoplifting_table = """
        CREATE TABLE crime (
            ID INT AUTO_INCREMENT PRIMARY KEY ,
            PREDICTION varchar(255), 
            CREATION_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Date date,
            CameraID varchar(255),
            LOCATION varchar(255),
            FilePath varchar(255)

        );
    """

        pw="1234"
        db="Criminal_Activity_Detection"
        connection = create_db_connection("localhost", "root", pw, db) # Connect to the Database
        execute_query(connection, create_shoplifting_table) # Execute our defined query
        
        ##populating the table
        pop_shoplifting = f"""
                            INSERT INTO crime (PREDICTION,Date, CameraID,LOCATION,FilePath) 
                            VALUES ("{label}",CURDATE(),"{text_input1}", "{text_input}","shooting");
                        """
        

        connection = create_db_connection("localhost", "root", pw, db)
        execute_query(connection, pop_shoplifting)
        print(pop_shoplifting)
        data= pd.read_sql('SELECT PREDICTION,CREATION_TIME, CameraID,LOCATION,FilePath FROM crime', dbConnection)
        
        if "id_row" not in st.session_state:
                st.session_state["id_row"] = ''
                selected_rows = []
        else:
                selected_rows = (list(range(len(st.session_state["id_row"]))))

        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        gb.configure_side_bar() #Add a sidebar
            #gb.configure_selection(selection_mode="multiple", use_checkbox=True,pre_selected_rows=selected_rows)
            #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
        gridOptions = gb.build()

        grid_response = AgGrid(
                    data,
                    #gridOptions=gridOptions,
                    #data_return_mode='AS_INPUT', 
                    update_mode='MODEL_CHANGED', 
                    #update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=False,
                    theme='alpine', #Add theme color to the table
                    enable_enterprise_modules=True,
                    height=550, 
                    width='100%',
                    reload_data=False
                )

        data = grid_response['data']
        selected = grid_response['selected_rows'] 
        df = pd.DataFrame(selected)
        st.dataframe(df)
        connection.close() 

    elif app_mode == 'Search':
        with st.form(key='searchform'):
            nav1,nav2,nav3,nav4 = st.columns([4,3,2,1])

            with nav1:
                search_term = st.text_input("Crime")
            with nav2:
                location = st.text_input("Location")
            with nav3:
                searchdate = st.date_input("Date")
            with nav4:
                st.text("Search ")
                submit_search = st.form_submit_button(label='Search')


        if submit_search:
            if search_term=="" and location=="" and searchdate:
                st.success("You searched for crimes on {}".format(searchdate))

                query = f"""
                            SELECT PREDICTION,CREATION_TIME, CameraID,LOCATION,FilePath
                            FROM crime
                            WHERE  
                            Date = "{searchdate}"
                            ;
                        """
            elif search_term and location=="" and searchdate:
                st.success("You searched for {} on {}".format(search_term,searchdate))
                query = f"""
                            SELECT PREDICTION,CREATION_TIME, CameraID,LOCATION,FilePath
                            FROM crime
                            WHERE  
                            Date = "{searchdate}"
                            and PREDICTION = "{search_term}"
                            ;
                        """
            elif search_term=="" and location and searchdate:
                st.success("You searched for crimes in {} on {}".format(location,searchdate))
                query = f"""
                            SELECT PREDICTION,CREATION_TIME, CameraID,LOCATION,FilePath
                            FROM crime
                            WHERE  
                            Date = "{searchdate}"
                            and LOCATION = "{location}"
                            ;
                        """
            elif search_term and location and searchdate:
                st.success("You searched for {} in {} on {}".format(search_term,location,searchdate))
                query = f"""
                            SELECT PREDICTION,CREATION_TIME, CameraID,LOCATION,FilePath
                            FROM crime
                            WHERE  
                            Date = "{searchdate}"
                            and LOCATION = "{location}"
                            and PREDICTION = "{search_term}"
                            ;
                        """     
            getdata=pd.read_sql(query, dbConnection)
            print(getdata)
            #no. of results
            num_of_results=len(getdata)
            st.text("Showing {} results..".format(num_of_results))

            df = pd.DataFrame(getdata)
            #print(df)
            st.dataframe(df)
