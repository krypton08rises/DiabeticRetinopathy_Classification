# DiabeticRetinopathy_Classification
Using binary classification to predict disease severity in Diabetic Retinopathy patients. 

The dataset has been obtained from the Kaggle DR challenge. 
This Repository will feature 2 models, a binary classifier to predict proliferate and severe categories of the original competition and a multi class classifier to detect 
mild, moderate and no DR categories respectively. 

Started with a VGG16 model, of 3 types
1. Pretrained on imagenet (__train.py__)
2. Untrained and randomly initialised (__model.py__)
3. Fine tuned with 10-12 frozen layers  (__ft_vgg.py__) 

Format of dataset(normalised)
->dataset
	->train	
		->0
		->1
	->val
		->0
		->1

## Preprocessing : 	

- The initial dataset consisted of various color fundus images of varying resolutions which are too large to train on (1050Ti) a smaller gpu so they have been downscaled them to 600*400. 
- The retina image normalisation code is present in preprocessing_fundus and scale_and_normalise.py respectively. These scripts were used by the kaggle dr challenge winners. 
- But instead of just normalising, we first use a technique called Constrast limited adaptive histogram equalisation, run (__preprocessing_fundus.py__). 
- The resulting images are much more suited for classification purposes.These images can now be normalised by running (__scale_and_normalise.py__)
### Augmentation :
- Only part of the dataset is augmented, namely the severe and proliferate categories(total ~1500 images). 
- The dataset is augmented, using only simple rotation(horizontal, vertical, both), and brightening(/diming).
- The resulting dataset has 7500 additional images (1500*5)

## Experiment : 
__Setup__ 

- An imagedatagenerator is used to pass the images to the network. There is no rescaling, or augmentation in the data generator. 
- Class weights are setup because of the imbalanced nature of the dataset. 
- The dataset contains all the original images from the kaggle DR dataset + 7500 augmented images.
- There are 2 experimental setups, one with only the Clahe images and one with the Clahe+Normalised images
- The last layer of the VGG16 is replaced by a dense layer mapping out to a softmax function.
- Training occurs in 2 steps,
1. Transfer Learning of the final layer with original imagenet weights intact. 
2. Fine Tuning with the first 3 layers frozen and the rest of the network retrained for 35 epochs.


## Results :
The model was trained for 35 epochs and it reached an f1score of 99% on the training set and 80% on validation set. The area under the ROC curve came out to be 0.87 and the model scored an F1 Score of 0.77!
