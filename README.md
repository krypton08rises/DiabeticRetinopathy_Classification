# DiabeticRetinopathy_Classification
Using binary and multiclass classification to predict disease severity in Diabetic Retinopathy patients. 

The dataset has been obtained from the Kaggle DR challenge. 
This Repository will feature 2 models, a binary classifier to predict proliferate and severe categories of the original competition and a multi class classifier to detect 
mild, moderate and no DR categories respectively. 

Started with a VGG16 model, 1 pretrained on imagenet and the other freshly trained on our dataset. 


VGG16_pretrained was relatively easy to train, reaching 82% accuracy in 50 epochs of the entire batch of ~34000 images. But it was still giving 
validation accuracy of just 62% because it had not yet learned to predict the severe categories of the dataset, since most of the images (~28000)
were of the mild or no DR category. Additionally, the smaller resolution of just (224, 224) which was a loss of a factor of 100 pixel density 
was contributing to this adaptation of the most important features of the retina images. Hence there was a huge requirement of preprocessing and 
completely removing the 


Specificity measured in the model - 0.9917355371900827 
Sensitivity measured in the model - 0.0759493670886076 --> clear indication of poor performance while determining severe DR cases


