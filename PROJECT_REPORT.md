# Brief Project Report

Group 19 developed a binary image classifier that distinguishes leopards from
tigers. Images were obtained from the Big Cats Image Classification Dataset on
Kaggle. Invalid and duplicate files were removed before stratified training,
validation and test sets were created. A MobileNetV3Small transfer learning
model was trained and evaluated in Google Colab, saved in Keras format and
connected to a Streamlit web application. To use the application, the user
uploads a clear JPG, PNG or WEBP wildlife image and presses **Run Wildlife
Analysis**. The application displays the predicted animal, confidence score and
both class probabilities. Challenges included similar coat patterns, varied
backgrounds, overfitting and unsupported images. Data augmentation, early
stopping, fine tuning, an untouched test set and an uncertainty message helped
to reduce these problems. Future work should add an independent validator for
rejecting images that contain neither animal.

**Word count:** 138  
**GitHub repository:** ADD_GITHUB_REPOSITORY_URL_HERE  
**Streamlit application:** ADD_STREAMLIT_URL_HERE
