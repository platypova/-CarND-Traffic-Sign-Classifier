# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_examples/signs_example.png "Example of sign form dataset (initial and grayscaled)"
[image2]: ./my_examples/signs_distribution.png "Distribution of signd types in the dataset"

[image3]: ./my_examples/german_traffic_signs/12.png "Traffic Sign 1"
[image4]: ./my_examples/german_traffic_signs/13.png "Traffic Sign 2"
[image5]: ./my_examples/german_traffic_signs/14.png "Traffic Sign 3"
[image6]: ./my_examples/german_traffic_signs/17.png "Traffic Sign 4"
[image7]: ./my_examples/german_traffic_signs/25.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

--
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy, csv and pickle libraries to download the dataset and transform it to a appropriate form for the further research.
Here are some statistic for the dataset:

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Here is a bar chart showing how signs of different types (according to labels) are distributed in training, testing and validation dataset:
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color information is not necessary for sign detection.

Here is an example of a traffic sign image before and after grayscaling.
![alt text][image1]

As a last step, I normalized the image data in order to prevent loss of information that can be caused by high-contrast pixels. I should add, that I've compared results of training neural network on normalized and not normalized data, and the accuracy was remarkably higher for the normalized data.

The preprocessing of the images is done by function 'preprocess_data' defined in codecells under heading "Pre-Process the Data Set ..." in jupyter notebook "Traffic_Sign_Classifier.ipynb".

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x24 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x48	    |
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x48 				    |
| Flatten				| 3D -> 1D     									|
| Fully connected		| 192 outputs  									|
| RELU					|												|
| Fully connected		| 120 outputs  									|
| RELU					|												|
| Fully connected		| 43 outputs  									|
| Output				| Logits       									|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001

I used the introduced dataset after preprocessing. 
The weights were initialized with a normal distribution function with zero mean and 0.01 for the standard deviation.
During training kepp probability for droput layers was chosen as 0.9.
Adam optimizer was chosen.
Changes of hyperparameters showed the influence on the accuracy. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9% 
* validation set accuracy of 94.8% 
* test set accuracy of 94.1%?

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
I've chosen the LeNet as a well-known and recommendated as a start point. It is simple to undertsand and use.

* What were some problems with the initial architecture?
The accuracy I got in the initial variant was not enough.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I've tried to modify existing layers and add additional layers. Finally I've got 6 layers: 3 convolutional and 3 fully connected.
I've also changed number of filters for convolutional layers, and it allowed me to increase train and validation accruacy.
I've also added dropout to avoid overfitting and it seems to help to get higher test accuracy. The keep probability was 0.9 during training, and its decreasing didn't show any better result.
Changing of padding didn't give remarkable effect also.
The final validation and test accuracy is enough, but I'm sure it can be improve, although it was interesting to modify a famous solution.

* Which parameters were tuned? How were they adjusted and why?
I've changed the number of epochs to 20, because 10 wasn't enough for my solution.
Batch size changing showed remarkable influence on accuracy, but in the end 128 was the best size.
Learning rate chaning also showed the influence, but the number of 0.001 was the best.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think that several convolution layers with quite big number of filters can help to obtain good results. Of course, all value have some limitations, and from some moment increase would make bad effect.
I noticed that droupout can help to avoid overfitting, although it shouldn't be applied to each layer, because it also may lead to low accuracy in the end.
Adding to much layers to architecture close to LeNet also may lead to bad results, but adding some (not many) may cause good effect.

If a well known architecture was chosen:
* What architecture was chosen?
So, I've chosen LeNet and modified it.
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		    | Yield     									| 
| No entry     			| No entry 										|
| Road work				| Road work										|
| Stop	      		    | Stop      					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 
However, I guess that those images aren't the most difficult for classification. But the solution managed to work on them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the codecells of the Ipython notebook below the heading Step3...
Firstly rthe dataset is loaded and transformed into appropriate form.
Then the predictions are made.

For the fifth image, the model is sure that this is yeild sign (probability of more than 99.9%), and the image does contain a yield sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 99.9999761581 %   	| Yield   								| 
| 0.0000124832 %    	| Keep right 							|
| 0.0000079815 %		| No vehicles							|
|  0.0000046151 %		| Speed limit (50km/h)					|
    | 0.0000001180 %	    | End of no passing      			|

For the other images the highest probability was almost 100%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


