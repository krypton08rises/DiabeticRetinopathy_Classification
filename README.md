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
The resulting images are much more suited for classification purposes.These images can now be normalised by running (__scale_and_normalise.py__)
- There is potential to crop these images and further drop the resolution to 400*400.


## Experiments : 

An imagedatagenerator is used to pass the images to the network. No augmentation is used since the dataset is already large enough. 

__Before Pretraining__ 
VGG16_pretrained was relatively easy to train, reaching 82% accuracy in 50 epochs of the entire batch of ~34000 images. But it was still giving 
validation accuracy of just ~62-65% because it had not yet learned to predict the severe categories of the dataset, since most of the images (~28000)
were of the mild or no DR category. Additionally, the smaller resolution of just (224, 224) which was a loss of a factor of 100 pixel density 
was contributing to this adaptation of the most important features of the retina images. Hence there was a huge requirement of preprocessing and 
completely removing the 


_Specificity_ - 0.9917355371900827 
_Sensitivity_ - 0.0759493670886076 --> clear indication of poor performance while determining severe DR cases
This can also be seen in the Auc curve showing that our model is a no skill model, and has not yet learned the differentiating features of severe and no DR retina images. 


__After Pretraining__

Class_weights are used in proportion to the imbalance in the two categories ~(27k for 0 and 8k for 1). 
The image data generator uses preprocessing function for all cnn's in keras applications. No need to rescale pixels value to 1/255
