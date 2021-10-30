# yolo_v3_rabbits
This project contains my implementation of the YoloV3 algorithm.

For this project, I used the VGG as a classifier, since it takes images of the size (224, 224, 3) and outputs tensors of the size (7, 7, 512). At some point it also outputs a tensors of sizes (14, 14, 256) and (28, 28, 128), which makes it an good candidate for YoloV3 model.

Some info about project files:
- data.py contains a class for data preparations and decoding model predictions;
- yolo_loss.py contains the loss class that I used to train the model;
- yolo_model.py contains a class for creating and training(and some other things) the model;
- train_yolo.py shows the code I used to train the model;
- kmeans.py contains functions for finding cluster centers for anchors;
- timeit.py calculates the time needed to predict the output for whole the training set(1980 images);
- test_images.py visualizes predictions;
- train_logs.txt contains training information;
- constants.py contains ... constants;
- utils/utils.py contains several useful functions.

Results:
The model mostly detects objects and gives it a good bounding box, but it also gives to much false positive predictions.
Precision ~ 0.1, Recall ~ 0.7.

There are several possible reasons for this:
- the original training set contains only 66 images (after the addition - 1980 images), which is not enough to train the exact model;
- My loss isn't good enough. Yolo Papers do not cover the case with only 1 class. Detecting only 1 class means that if the object is detected then it is 100% rabbit and the model will always predict 100% for that class (because it only calculates losses for the boxes where the object is). I tried to modify it and calculate losses for all boxes, but in this case the model always gives 0% for the class. It looks like the inference of the class is too sparse (only one 100% class prediction for 1029 boxes, and that's with only 1 anchor!). So I tried to add a ratio between class predictions for boxes with and without an object, but it doesn't seem to work well enough (maybe this will fix the problem for a while, but it's not robust). As a result, there are too many false positives.
