import speech_recognition as sr 
r = sr.Recognizer() 
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source) 
def get_transcript():
    mic = sr.Microphone()
    with mic as source: 
        try: 
            audio = r.listen(source, timeout=5, phrase_time_limit=10) 
            try :
                result = r.recognize_google(audio)
            except :
                return None
        except:return None 
    return result