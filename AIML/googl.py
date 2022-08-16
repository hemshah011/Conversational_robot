from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sys
from time import sleep

if sys.argv[2] == "1":

	url = "https://google.com/search?q="
	for i in range(1,len(sys.argv)-1):
		url+=sys.argv[i]+" "
elif sys.argv[2] == "2":
	print("Here are the pictures")
	url = "https://www.google.com/search?tbm=isch&q="
	for i in range(1,len(sys.argv)-1):
		url+=sys.argv[i]+" "
try:
	driver = webdriver.Chrome('bot\chromedriver_win32 (1)\chromedriver.exe')
	driver.get(url)
	sleep(10)
	driver.quit()
except:
	print("Cannot utilise this functionality for this device")