# Environment_Sound_Classifier

The project is built to classify environment sounds in real-time, by taking intermittent audio input of 5 second samples and processing them in mutliple threads.

Dataset used for this project was ESC-10.

ESC-10 is a part of ESC dataset (Environmental Sound Classification).


The ESC dataset is a collection of short environmental recordings available in a unified format (5-second-long clips, 44.1 kHz, single channel, Ogg Vorbis compressed @ 192 kbit/s). All clips have been extracted from public field recordings available through the Freesound.org project. Please see the README files for a detailed attribution list. The dataset is available under the terms of the Creative Commons license - Attribution-NonCommercial.

The dataset consists of three parts:

ESC-50: a labeled set of 2 000 environmental recordings (50 classes, 40 clips per class),
ESC-10: a labeled set of 400 environmental recordings (10 classes, 40 clips per class) (this is a subset of ESC-50 - created initialy as a proof-of-concept/standardized selection of easy recordings),
ESC-US: an unlabeled dataset of 250 000 environmental recordings (5-second-long clips), suitable for unsupervised pre-training.

Snapshots of Results:

![](https://user-images.githubusercontent.com/35894429/51501548-5532d380-1df8-11e9-93fa-be5d8ed63baa.png)

![](https://user-images.githubusercontent.com/35894429/51501588-86130880-1df8-11e9-8d33-9e7653229426.png)

![](https://user-images.githubusercontent.com/35894429/51501602-9d51f600-1df8-11e9-8212-57fa5a92c232.png)
