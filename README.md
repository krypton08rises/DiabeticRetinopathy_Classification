# DiabeticRetinopathy_Classification
Using binary classification to predict disease severity in Diabetic Retinopathy patients. 

The dataset has been obtained from the Kaggle DR challenge. 
This Repository will feature 2 models, a binary classifier to predict proliferate and severe categories of the original competition and a multi class classifier to detect 
mild, moderate and no DR categories respectively. 

Started with a VGG16 model, of 3 types
1. Pretrained on imagenet 
2. Untrained and randomly initialised
3. Fine tuned with 10-12 frozen layers. 


Hence we move onto the next phase of the problem, cropping and image normalization to highlight the differentiating factors. 


## Preprocessing : 

- The initial dataset consisted of various color fundus images of varying resolutions which were too large to train on my gpu(1050Ti) and so I downscaled them to 600*400. 
- The retina image normalisation code is present in preprocessing_fundus and scale_and_normalise.py respectively. These scripts were used by the kaggle dr challenge winners. 
- But instead of normalising, I have used Constrast limited adaptive histogram equalisation. 
The resulting images are much more suited for classification purposes.
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
To remove the misleading accuracy metric, class_weights are used in proportion to the imbalance in the two categories ~(27k for 0 and 8k for 1). 