# handshape-detection
A pipeline for detecting hands and recognise static handshape.
There are two sub networks:
The first is an SSD model with a mobilenet backbone, used for the detection of the hand. The pytorch implementation is provided by qfgaohao from here (https://github.com/qfgaohao/pytorch-ssd).
The Second sub network is a vgg16 architecture, that take the cropped detected image from the first sun network and classify the handshape. The model is trained to detect 24 static handshape (ASL alphabet).
