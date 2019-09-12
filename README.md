# Raspberry-survelillance-using-opnecv
Project for using a raspberry pi with pi camera as a surevillance system to alert owner if any unknown user enters the room and sends owner a video clipping for duration of entry highlighting faces via email 
We make use fo opencv's dnn models for our purpose

## Requirements 
For the pupose of this project you will need opencv ,scikit-learn imutils for training the dataset and perfroming the faceal recognition

```bash
pip install opencv-python
pip install imutils
pip install scikit-learn

```
 
models:

 '''bash
 openface_nn4.small2.v1.t7
 
 ''''
  A torch moidel you will need for producing 128-D embeddings
You will also need another caffe model for performing the facial recginiton

'''bash
res10_300x300_ssd_iter_140000.caffemodel

'''
and

'''bash
deploy.prototxt"

'''

Pickle is also used to store facial embeddings generated from applying torch models on images

## Steps to run surveillance

STEP#1 Extract embeddings from face dataset
give path to different data sets and files locations
'''bash
python extract_embeddings.py --dataset images --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

'''

STEP#2 Train face recognition model

'''bash
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

'''

STEP#3 Run surveillance 

'''bash
python raspberrypi_surveillance.py

'''
 
## License
[MIT](https://choosealicense.com/licenses/mit/)
