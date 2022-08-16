import sys
import argparse
sys.path.append("Response Generation/")

# Getting the calling function
parser = argparse.ArgumentParser(description = "The bot.")

parser.add_argument("-m", "--mode", default = "msg", choices = ['msg', 'trigger'],  help = "Mode of execution : Message box/ Trigger word detection")
args = parser.parse_args()
func = args.mode


print("Loading dependencies ...")
import aiml
kernel=aiml.Kernel()
kernel.learn("AIML/startup.xml")
kernel.respond("load aiml b")
import TextToSpeech.tts as tts
import SpeechRecognition.gstt_real_time as stt
import numpy as np
from response_generation import *
import tkinter as tk
import speech_recognition as sr
from tkinter import messagebox
import time
print("Vatican Cameos.")


def msg_box():
    root = tk.Tk()
    canvas1 = tk.Canvas(root, width = 300, height = 300)
    canvas1.pack()
    root.withdraw()
    msgbox = tk.messagebox.askquestion("ConvoBot", "Do you want to start the conversation?")
    if msgbox == "yes":
        root.destroy()
        return True
    else:
        root.destroy()
        return False


def trigger_word_detection(trigger = "hello", kill = "bye"):
    print("Waiting for triggers.")
    start = time.time()
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        r.adjust_for_ambient_noise(source)
    while True: # Listens in 10 second cycles
        try:
            with m as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                command = r.recognize_google(audio)
                command = command.split()[0]
            if command == trigger:
                return True
            elif command == kill:
                return False
        except:
            command = None
            print(f"\rHave been listening for {time.time() - start} seconds  ", end = "")
            continue
    return

def converse():
    done = False

    glomar = ['I didn\'t get that', 'You\'ll have to be a little louder.', 'You are not audible.', 'I am afraid I didn\'t get that.']

    while not done:
        print("Say something.")
        stimulus = stt.get_transcript()
        # Un-comment for mic-less tests
        # if stimulus is None:
        #    stimulus = input()

        if stimulus is not None:
            
            if stimulus.lower() == "bye":
                done = True
            print("You said : ", stimulus)

            if kernel.respond(stimulus):
                response=kernel.respond(stimulus)
            else:
                response = get_response(stimulus)
        
        else :
            response = np.random.choice(glomar)
        tts.play_response(response)
        print("And I said : ", response)

def main():
    if func == "msg" : call_func = msg_box
    else: call_func = trigger_word_detection
    
    
    while True:
        inclination = call_func()
        if not inclination:
            break
        converse()

main()
        
