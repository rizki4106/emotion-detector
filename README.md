# Emotion Detector

Implementation of `Convolutional Neural Network` for recognize face emotion. we've train hundred times this model to get better accuracy.

<img src="output/output2.jpg">
<small>image source : </small>
<small><i>https://www.boisestate.edu/news/2019/03/18/boise-state-receives-delegation-from-taiyuan-institute-of-technology/</i></small>

## How to use

```bash
git clone https://github.com/rizki4106/emotion-detector.git
```

```bash
cd emotion-detector && pip install tensorflow opencv
```

```python
# main.py

from src.detector import DetectEmotion

# if gui true when you run this script
# it will show the gui
# if gui false you can't see anything but
# you can see the output just print it.

alg = DetectEmotion(gui=True)

# takes 2 arguments
# 1. file -> file path
# 2. media_type -> type of media ( image or video)
# for now just support image

res = alg.predict("img/team2.jpeg", "image")
print(res)
```
This models can predict 7 type of face emotion

| name | label |
| ---- | ----- |
| angry | 0 |
| disgusted | 1 |
| fearful | 2 |
| happy | 3 |
| neutral | 4 |
| sad | 5 |
| suprised | 6 |

## About Data
Thanks to [ananthu017](https://www.kaggle.com/ananthu017/) for providing emotion dataset on [kaggle](https://www.kaggle.com/ananthu017/emotion-detection-fer). We've train this model using those data.