# Brief Project Report

The Group 19 project developed a binary image classifier for distinguishing a
leopard from a tiger. Images were obtained from the public Big Cats Image
Classification Dataset on Kaggle, after which only the leopard and tiger
classes were retained. Invalid and duplicate files were removed before
stratified training, validation and test splits were created. A
MobileNetV3Small transfer learning model was trained and evaluated in Google
Colab, then connected to a Streamlit web application. To use the application,
the user uploads a clear JPG, PNG or WEBP image and receives the predicted
animal and confidence score. The main challenges were limited class data,
similar coat patterns, varied backgrounds and possible overfitting. Data
augmentation, early stopping, fine tuning and an untouched test set reduced
these problems. Future improvement should add more verified field images,
perform external testing and include an unknown-animal rejection option.

**Deployed application:** ADD_STREAMLIT_URL_HERE  
**GitHub repository:** ADD_GITHUB_REPOSITORY_URL_HERE
