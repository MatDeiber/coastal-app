import streamlit as st
import datetime, time
from datetime import timedelta
from streamlit_folium import folium_static
import folium
import matplotlib.pyplot as plt
'''
# TaxiFareModel Prediction
'''
page_load_start_time = time.time()
page_load_time_placeholder = st.sidebar.empty()
'''
## Enter taxi ride information: .strftime("%Y-%m-%d %H:%M:%S")
'''
d = st.date_input("Enter your date of travel",
                  datetime.datetime(2022, 1, 1, 1))
st.write(d)
t = st.time_input('Choose a time', datetime.time(8, 45))
date_time = datetime.datetime.combine(d, t)
date_str = date_time.strftime("%Y-%m-%d %H:%M:%S")
st.write(date_time)
pickup_longitude = st.number_input('pickup longitude')
pickup_latitude = st.number_input('pickup latitude')
dropoff_longitude = st.number_input('dropoff longitude')
dropoff_latitude = st.number_input('dropoff latitude')
passenger_count = st.slider('Select a modulus', 1, 10, 2)
url = 'https://taxifare.lewagon.ai/predict'
data = {
    "pickup_latitude": pickup_latitude,
    "pickup_longitude": pickup_longitude,
    "dropoff_latitude": dropoff_latitude,
    "dropoff_longitude": dropoff_longitude,
    "passenger_count": int(passenger_count),
    "pickup_datetime": date_str,
}
import requests
if st.button('Predict'):
    # print is visible in the server output, not in the page
    response = requests.get(url, params=data)
    pred = response.json()['prediction']
    temp_date = []
    for i in range(10):
        temp_date.append(date_time + (i - 5) * timedelta(minutes = 15))
    """
    Result is:
    """

    predictions = []
    for dt in temp_date:
        data = {
            "pickup_latitude": pickup_latitude,
            "pickup_longitude": pickup_longitude,
            "dropoff_latitude": dropoff_latitude,
            "dropoff_longitude": dropoff_longitude,
            "passenger_count": int(passenger_count),
            "pickup_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        response = requests.get(url, params=data)
        predictions.append(response.json()['prediction'])



    st.write("It will cost $" + str(pred))

    fig, ax = plt.subplots()
    ax.plot(temp_date, predictions)
    ax.set_xlabel(time)

    st.pyplot(fig)




    # center on Liberty Bell
    m = folium.Map(location=[pickup_longitude, pickup_latitude], zoom_start=16)

    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker([pickup_longitude, pickup_latitude],
                  popup="Liberty Bell",
                  tooltip=tooltip).add_to(m)

    folium.Marker([dropoff_longitude, dropoff_latitude],
                  popup="Liberty Bell",
                  tooltip=tooltip).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)



else:
    st.write('Enter the params first')
page_load_duration = time.time() - page_load_start_time
page_load_time_placeholder.markdown(f'{round(page_load_duration, 3)} seconds')
