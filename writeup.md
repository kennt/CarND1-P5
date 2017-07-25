# Vehicle Finding

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image10]: ./output_images/vehicles-example.png
[image11]: ./output_images/non-vehicles-example.png

[image20]: ./output_images/hog1.jpg
[image21]: ./output_images/hog2.jpg
[image22]: ./output_images/hog3.jpg
[image23]: ./output_images/hog0.jpg

[image30]: ./output_images/hog-subsample1.jpg
[image31]: ./output_images/hog-subsample2.jpg
[image32]: ./output_images/hog-subsample3.jpg
[image33]: ./output_images/hog-subsample4.jpg

[image40]: ./bbox-example-image.jpg
[image41]: ./output_images/pipeline1.jpg
[image42]: ./output_images/pipeline2.jpg
[image43]: ./output_images/pipeline3.jpg

---

## 1. Histogram of Oriented Gradients (HOG)

#### 1. HOG Feature Extraction

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle example][image10]
![Non-vehicle example][image11]

The code for this step is contained in ```3. Feature Extraction``` in ```code.ipynb``` IPython notebook.  The code for extracting spatial bins, the histograms, and the HOG features is all included.  All of the features are brought together and concatenated in ```extract_features()```.

Here is an example visualization provided from the HOG function, the images are separated into three different channels (I'm using the YCrCb color space) (with the original image at the beginning).

![Original image][image23]
![Hog channel 1][image20]
![Hog channel 2][image21]
![Hog channel 3][image22]


The various choice of HOG parameters was chosen manually, running various experiments with the various test images.

I first looked into the effect of various color spaces. I first tried RGB and HSV.  But later tried YCrCb to see if I got better performance.

I also found that using all of the color channels as HOG features worked better than using a single color channel.

I ran GridSearchCV() to optimize for accuracy by looking at the parameter C.  Previous testing resulted in 'rbf' being a better classifier than 'linear'.

### 2. Sliding Window Search

#### 1. Implementation

The Sliding Window Search algorithm is implemented with HOG Subsampling. This is in the ```6. Sub-sampling HOG``` section of ```code.ipynb```. The parameter search was done with much exploration, trying to keep a good mixture of sizes to capture vehicles near and far.

I ended up with three scales.  Also the scales were sized so that the further rectangles would be searched more in the middle of the image, while the larger rectangles would be kept lower.  This helped to increase the speed of the algorithm as less of the image needed to run through the HOG feature generation.  The coverage is shown below in the four images.

The actual values (and code) that contains the actual ranges is in ```7. VehicleFinder Class``` in ```code.ipynb```. This class is used to implement the actual image processing needed to perform the video processing.

The actual subsampling can be found in lines 21-24.

![Hog subsample 1][image30]
![Hog subsample 2][image31]
![Hog subsample 3][image32]

#### 2. Samples and optimization

This is the original image.

![Original image][image40]


The next step is to use the sub-sampling windowed search to find the features.  This is this image, there are multiple matches at different sizes and are also overlapped.

![First step][image41]

We then generate a heatmap (by having each box add a point of heat).  In addition, values that are covered by 2 boxes have a point boost in order to keep the rectangles large enough (and thus keeping one large rectangle rather than many small rectangles).  This code is in ```7. Heatmap and Labeling```.

![Heatmap][image42]

From the heatmap, we can then draw the rectangles around the area of interest.

![Labels][image43]
---

## 2. Video Implementation

Here's a [link to my video result](./project_output.mp4)

### Filtering (false positives)

I used a heatmap (to generate a bounding box for overlapping detections).  These were merged using ```scipy.ndimage.measurements.label()```.  I averaged the heatmaps over 4 frames and thresholded the average heatmap to generate the final bounding boxes. This helped to remove random false positives (although some do pop up every now and then).

The code to maintain the frame averaging is in the ```8. VehicleFinder Class```, lines 5-9, 26-45.


---

## 3. Discussion

#### 1. Optimization
Trying to determine the proper values (given the many parameters) was difficult to discern. This was mostly trial-and-error.  Using the sub-sampling in the proper position in the image really helped with performance.

#### 2. Using the proper image values
This actually cause a lot of false starts and errors.  I would train the model with one parameter set and predict with a different set of parameters, which would lead to errors (or false results).

#### 3. Not enough examples
It would be interesting to see how this performs with the diverse vehicles in reality, such as trucks, trailers, etc...

#### 4. Proper example set
Due to the time series nature of some of the data, I found it was better to actually use a subset of the data for training.  I found that surprising (although I probably shouldn't have been).

#### 5. Right-side of edge is not examined
Looking at the images generated for the sliding window section, the right-edge of the frame is not looked at (which is why we lose tracking of the car on the right when it first enters and when it falls behind).  Maybe center the algorithm rather than starting from the edge or see if we can extend it all the way to the edge.

#### 6. Proper classifier
Using the RBM kernel SVM appears to be much better than using the LinearSVM.  Unfortunately I didn't have time to tune the gamma parameter also, but the results worked pretty well (better detection of the white car).

