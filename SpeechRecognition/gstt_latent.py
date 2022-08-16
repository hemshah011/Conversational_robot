# importing the module
import speech_recognition as sr
import time
r = sr.Recognizer()
audio_file = sr.AudioFile('/home/abhay/Projects/ConvoBot/Programming Assignments/Speech Recognition/easywaysout/download.flac')

def get_transcript(audio_file=audio_file):
  with audio_file as source:  
    audio = r.record(source)
  result = r.recognize_google(audio)
  return result

if __name__ == "__main__":
  print(get_transcript())