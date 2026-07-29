# Laboratory Exercise 10: Observations, Results and Discussion

## 1. Libraries and tools

The project imports TensorFlow for loading and running the trained neural
network, Streamlit for the web interface, NumPy for numerical array handling,
Pillow for opening and converting images, JSON for reading model settings and
Pathlib for reliable file paths. During training, Scikit-learn is used for data
splitting and evaluation metrics, while Matplotlib and Seaborn display learning
curves and the confusion matrix.

## 2. Saved model and tokenizer

The best model is saved as `leopard_tiger_model.keras`. ModelCheckpoint retains
the version with the best validation AUC. No tokenizer is required because the
task processes images rather than text. The class order, image size and
decision threshold are saved in `model_info.json`.

## 3. Streamlit application components

The application contains a title and project identity, instructions, supported
file information, an image uploader, an image preview, a prediction button, an
output card, class probabilities, a confidence score and an uncertainty
message. The sidebar summarises the task, model and correct input format.

## 4. Loading the model and input features

TensorFlow loads `leopard_tiger_model.keras` with compilation disabled because
the deployed application performs inference only. The input feature is one RGB
image. Pillow converts it to RGB, resizes it to 224 by 224 pixels and NumPy
places it in a batch before prediction.

## 5. User input

The required input is a clear JPG, JPEG, PNG or WEBP photograph containing one
leopard or tiger. Streamlit collects the image with `st.file_uploader`. Invalid
or unreadable files produce an error message.

## 6. Prediction button

The **Run Wildlife Analysis** button triggers prediction. The application
checks for the saved model, preprocesses the uploaded image and passes the
resulting batch to the trained network.

## 7. Tokenizer use

A tokenizer is not applicable because the project contains no text
classification or sequence-processing task.

## 8. Prediction results

The sigmoid output represents tiger probability. Leopard probability is one
minus tiger probability. The threshold in `model_info.json` determines the
selected class. The interface displays both probabilities and the selected
class confidence. Scores below the configured minimum confidence produce an
uncertainty message.

## 9. Streamlit and cloud deployment

The project is tested with `streamlit run app.py`. All files are uploaded
directly to the GitHub repository root. Streamlit Community Cloud is connected
to the correct GitHub account, the branch is set to `main`, and `app.py` is the
entrypoint. Dependencies are installed from `requirements.txt`.

## 10. Challenges and resolutions

The first deployment failed because the outer folder name contained spaces and
`(1)`, causing Streamlit to misread the requirements path. Uploading the project
contents directly to the repository root resolved the problem. Streamlit and
GitHub account linking also required reauthorization. During application
testing, unrelated images were still assigned one of the two animal classes
because the model is binary. A prediction button, clear usage notice and
uncertainty threshold were added. A separate supported-image validator remains
the recommended future improvement.
