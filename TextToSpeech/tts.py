from gtts import gTTS
import os
from playsound import playsound

def play_response(resp):
    tts = gTTS(text=resp, lang='en')
    filename = f'{resp}.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename) #remove temperory file

