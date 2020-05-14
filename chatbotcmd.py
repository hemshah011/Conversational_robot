import aiml
kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")
print("Hey,I am Jarvis your virtual assistant")
#crl+c to quit.
while True:
    print(kernel.respond(input("Here to help :")))
