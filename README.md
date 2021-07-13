# Conversational Robot

Robotics Club Summer Project 2020

## Mentors

* [Paras Mittal](https://github.com/PrsMittal)
* [Ashwin Shenai](https://github.com/ashwin2802)

## Team Members

* **Group A** - [Varun Khatri](https://github.com/varunk122), [Prateek Jain](https://github.com/Prateekjain09), [Adit Khokhar](https://github.com/adit-khokar), [Atharva Umbarkar](https://github.com/AtharvaUmbarkar), [Ishir Roongta](https://github.com/isro01)
  * Group Repo - [https://github.com/isro01/Conv_bot]</br></br>
* **Group B** - [Shiven Tripathi](https://github.com/ShivenTripathi), [Prakhar Pradhan](https://github.com/prakhariitk), [Mohd Muzzammil](https://github.com/XaltaMalta), [Sidhartha Watsa](https://github.com/sidwat), [Azhar Tanweer](https://github.com/Azhar-999-coder)
  * Group Repo - [https://github.com/conversational-robot]</br></br>
* **Group C** - [Ambuja Budakoti](https://github.com/AmbujaBudakoti27), [Devansh Mishra](https://github.com/devansh20), [Hem Shah](https://github.com/hemshah011), [Kavya Agarwal](https://github.com/KavyaAgarwal2001), [Preeti Kumari](https://github.com/preeti1428)
  * Group Repo - [https://github.com/AmbujaBudakoti27/ConversationalRobot]</br></br>
* **Group D** - [Abhay Dayal Mathur](https://github.com/Stellarator-X), [Amitesh Singh Sisodia](https://github.com/Amitesh163), [Anchal Gupta](https://github.com/anchalgupta05), [Arpit Verma](https://github.com/Av-hash), [Manit Ajmera](https://github.com/manitajmera), [Sanskar Mittal](https://github.com/sanskarm)
  * Group Repo - [https://github.com/Amitesh163/ConvBot_group]
  
***

## Aim

The aim of this project was to make a **Talking bot**, one which can pay attention to the user's voice and generate meaningful and contextual responses according to their intent, much like human conversations.

## Ideation

This project was divided into overall three parts :

* [Speech to Text conversion](https://github.com/Amitesh163/ConvBot_group/tree/master/SpeechRecognition) <sup>[1]</sup>
* [Response Generation](https://github.com/Amitesh163/ConvBot_group/tree/master/Response%20Generation) <sup>[2]</sup>
* [Text to speech conversion](https://github.com/Amitesh163/ConvBot_group/tree/master/TextToSpeech)

## Overall Pipeline of the Project

![overall pipeline](images/speech.png)

### Speech Recognition

We used *google-speech-to-text (gstt)* API for the conversion of speech to text transcripts with a WER(Word Error Rate) of *4.7%*.

### Response Generation

We used a subset of the OpenSubtitles <sup>[4]</sup> dataset to train our response generation model, which was a combination of Context-based and Topic-based Attention Model.</br>
This model has an encoder network which produces context vector for an input sentence followed by an attention mechanism which decides how much attention is to be paid to a particular word in a sentence and finally a decoder network which uses attention weights and context vectors to generate words of the output sentence i.e. response. We also added an [AIML pipeline](AIML) to our model for responding to some specific pattern of inputs which include greetings, emotions, jokes etc and also added Weather Forecasting and Googling capabilities.</br>
Some of the output examples that we've produced with our model are:</br>

<p align = "center">
<img src="images/responseexamples.jpeg">
</p>

### Text to speech conversion

We used the *google-text-to-speech (gtts)* API for the conversion of text transcripts of responses back to speech.</br>
The API uses *playsound* to play a temporary mp3 file created from the model's textual response.

***

## Usage

Install the required dependencies :

```bash
$pip install -r requirements.txt
$sudo apt-get install gstreamer-1.0
```

Training checkpoints, LDA model weights and tokens can be found [here](https://drive.google.com/drive/folders/18o-bFpJjy1S4UHUbdTjQnb2B_IK4bIM5?usp=sharing)

Required File Structure:

```txt
Response Generation
├── bin
│   ├── LDA
│   ├── Tokens.txt
│   ├── topic_dict.dict
│   ├── training_checkpoints
│   └── glove.42B.300d.txt
└── ...
```

### Running the bot

```bash
usage: bot.py [-h] [-m {msg,trigger}]

The bot.

optional arguments:
  -h, --help            show this help message and exit
  -m {msg,trigger}, --mode {msg,trigger}
                        Mode of execution : Message box/ Trigger word
                        detection
```

#### Modes

* **Message Box** - Provides a GUI for the user to start the conversation at the click of a button.
* **Trigger Word Detection** - The program listens in the background and starts the conversation upon hearing the trigger word.
  * Commencement Trigger - _Hello_
  * Concluding Trigger - _Bye_

### Functionality

* Casual Conversations
* Google search along with an explicit search feature for images
* Weather Information

***

### Demonstration

The video demonstration of this project can be found [here](https://drive.google.com/file/d/1jAmxwfUnrx9qa9nh8Sol4ZByIH_w7YRE/view?usp=drivesdk).

***

## References

1. _Deep Speech 2: End-to-End Speech Recognition in English and Mandarin_
   * **Link** : [https://arxiv.org/abs/1512.02595]
   * **Author(s)/Organization** : Baidu Research – Silicon Valley AI Lab
   * **Tags** : Speech Recognition
   * **Published** : 8 Dec, 2015

2. _Topic Aware Neural Response Generation_
   * **Link** : [https://arxiv.org/abs/1606.08340]
   * **Authors** : Chen Xing, Wei Wu, Yu Wu, Jie Liu, Yalou Huang, Ming Zhou, Wei-Ying Ma
   * **Tags** : Neural response generation; Sequence to sequence model; Topic aware conversation model; Joint attention; Biased response generation
   * **Published** : 21 Jun 2016 (v1), 19 Sep 2016 (v2)

3. _Topic Modelling and Event Identification from Twitter Textual Data_
   * **Link** : [https://arxiv.org/abs/1608.02519]
   * **Authors** : Marina Sokolova, Kanyi Huang, Stan Matwin, Joshua Ramisch, Vera Sazonova, Renee Black, Chris Orwa, Sidney Ochieng, Nanjira Sambuli
   * **Tags** : Latent Dirichlet Allocation; Topic Models; Statistical machine translation
   * **Published** : 8 Aug 2016

4. _OpenSubtitles_ (Dataset)
   * **Link** : [http://opus.nlpl.eu/OpenSubtitles-v2018.php]
