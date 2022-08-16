# Conversational Robot

Robotics Club Summer Project 2020

## Mentors

* [Paras Mittal](https://github.com/PrsMittal)
* [Ashwin Shenai](https://github.com/ashwin2802)

## Team Members

* **Group** - [Ambuja Budakoti](https://github.com/AmbujaBudakoti27), [Devansh Mishra](https://github.com/devansh20), [Hem Shah](https://github.com/hemshah011), [Kavya Agarwal](https://github.com/KavyaAgarwal2001), [Preeti Kumari](https://github.com/preeti1428)
  * Group Repo - [https://github.com/AmbujaBudakoti27/ConversationalRobot]</br></br>
  
***

## Overall Pipeline of the Project

The main aim of this project was to make a conversation bot able to take in audio input and output a meaningful reply keeping in mind factors like context ant intent in the input given by user.

The three main parts of this project were:

1. Speech to text
2. Topic attention (to generate a response)
3. Text to speech

### CTC_MODEL

This model is implemented to convert the audio messages of the user into text.
<p align="center">
<img src="https://user-images.githubusercontent.com/56124350/85904215-c2386d00-b825-11ea-99cf-b635187e99cc.png"     width="700" height="200">
</p>

Dataset opted for training: [Librespeech](http://www.openslr.org/12/)
### ENCODER - DECODER MODEL

This model is implemtented to cover the response generation part of the conversational bot.
We trained this model on the dataset [Opensubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)

<p align = "center">
  <img src="https://user-images.githubusercontent.com/56124350/85904325-0c215300-b826-11ea-9312-e8ccd9cb2ce1.png" width="400" height="458">
  <img src="https://user-images.githubusercontent.com/56124350/85904328-0e83ad00-b826-11ea-9f48-179de5c00319.png" width="400" >
</p>

### LDA MODEL 

This model is implemented to add topic awareness to ENCODER - DECODER Model for better response generation by focusing it's "attention" to only specific parts of the input rather than the whole sentence.

### Optimal Number of Topics

This graph shows the optimal number of topics we need to set for news articles dataset.
<p align = "center">
  <img src="https://user-images.githubusercontent.com/56124350/85904664-f2ccd680-b826-11ea-8ba2-09607478d22e.png" width="500" height="200">
</p>

### Gensim LDA Model parameters

* **corpus** —   Stream of document vectors or sparse matrix of shape (num_terms, num_documents) <
* **id2word** – Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.
* **num_topics** — The number of requested latent topics to be extracted from the training corpus.
* **random_state** — Either a randomState object or a seed to generate one. Useful for reproducibility.
* **update_every** — Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.
* **chunksize** — Number of documents to be used in each training chunk.
* **passes** — Number of passes through the corpus during training.
* **alpha** — auto: Learns an asymmetric prior from the corpus
* **per_word_topics** — If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature-length (i.e. word count)

### Text to Audio

[gTTS](https://pypi.org/project/gTTS/), a python library was used to make a function to output audio from the generated responses.
## Usage

Install the required dependencies :

```bash
$pip install -r requirements.txt
$sudo apt-get install gstreamer-1.0
$python3 -m spacy download en
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
