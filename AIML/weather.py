import os
import json
import sys
from datetime import date, timedelta

try:
    from darksky import forecast
    import requests
except:
    print("This functionality is not currently available for your device")
    sys.exit()
API_KEY="58ce87e229a6183b47a82b013ead67b6"
url = 'http://photon.komoot.de/api/?q='
def weather(address,query):

    resp=requests.get(url=url+address)
    data=json.loads(resp.text)
    latitude  = data['features'][0]['geometry']['coordinates'][1]
    longitude = data['features'][0]['geometry']['coordinates'][0]
    answer    = forecast(API_KEY,latitude,longitude)

    if query=='temperature':
        return int((5.0/9)*(answer['hourly']['data'][2]['temperature']-32))
    elif query=='humidity':
        return(answer['hourly']['data'][2]['humidity'])
    else:
        return(answer['hourly']['data'][2]['summary'])
#print(str(weather('london',"temperature"))+" degrees C")
if sys.argv[2] == "1":
    if len(sys.argv)<=2:
        print(str(weather(sys.argv[1],"temperature"))+" degrees C")
    else:
        print(str(weather(sys.argv[1]+" "+sys.argv[2],"temperature"))+" degrees C")
elif sys.argv[2] == "2":
    if len(sys.argv)<=2:
        print(str(weather(sys.argv[1],"humidity")))
    else:
        print(str(weather(sys.argv[1]+" "+sys.argv[2],"humidity")))
else:
    if len(sys.argv)<=2:
        print(str(weather(sys.argv[1],"summary")))
    else:
        print(str(weather(sys.argv[1]+" "+sys.argv[2],"summary")))
