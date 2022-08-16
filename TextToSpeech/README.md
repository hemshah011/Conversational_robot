# Text To Speech

Installing Dependencies:

```bash
$ pip3 install -r requirements.txt
$ sudo apt-get install gstreamer-1.0
```

Testing the module:

```bash
$ python3 -i tts.py
>>> play_response("<String>")
```

Usage:

```python
...
from tts import *
... # Get Response String st
play_response(st)
...
```