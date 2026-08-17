# GET 324 Group 19: Leopard versus Tiger

This project completes Laboratory Exercise 10 with a trained binary
image model and a wildlife-themed Streamlit interface called **WildSpot**.

## Main improvements

- Unique responsive wildlife interface
- Explicit prediction button
- Uploaded-image preview
- Confidence and both class probabilities
- Uncertainty message based on a configurable threshold
- Clear warning about unsupported images
- Discussion-question answers
- Submission checklist
- Safe root-level GitHub structure

## Project files

```text
get324-group19-leopard-tiger/
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_app_helpers.py
├── app.py
├── inference_utils.py
├── Group19_Leopard_vs_Tiger_Colab.ipynb
├── leopard_tiger_model.keras
├── model_info.json
├── CONTRIBUTORS.md
├── DISCUSSION_ANSWERS.md
├── PROJECT_REPORT.md
├── SUBMISSION_CHECKLIST.md
├── README.md
└── requirements.txt
```

The ZIP does not contain `leopard_tiger_model.keras`. The Colab notebook
generates the model after training.

## 1. Train the model in Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook**.
3. Upload `Group19_Leopard_vs_Tiger_Colab.ipynb`.
4. Select **Runtime > Change runtime type > T4 GPU**.
5. Run every cell from top to bottom.
6. Follow the Kaggle authentication prompt if one appears.
7. Inspect the class counts and sample images.
8. Record accuracy, precision, recall, F1 score and confusion matrix.
9. Download:
   - `leopard_tiger_model.keras`
   - `model_info.json`
10. Put both files beside `app.py`. Replace the placeholder JSON file with the
    downloaded Colab version.

## 2. Preserve the uncertainty setting

The placeholder `model_info.json` contains:

```json
"minimum_confidence": 0.8
```

The original Colab-generated JSON may not include this field. The rebuilt app
automatically uses `0.8` when the field is absent, so deployment will still
work. You may also add it manually after downloading the Colab file.

## 3. Upload correctly to GitHub

Create or update the repository:

```text
get324-group19-leopard-tiger
```

Open the extracted project folder and upload its **contents**, not the outer
folder. GitHub's first repository page must show:

```text
app.py
requirements.txt
model_info.json
leopard_tiger_model.keras
```

Avoid folder names containing spaces or `(1)`.

## 4. Deploy with Streamlit

Use the following settings:

```text
Branch: main
Main file path: app.py
Python version: 3.11
```

Streamlit installs the packages from `requirements.txt`. If the app already
exists, upload the rebuilt files to the same repository and reboot it. If the
entrypoint still points to an old nested folder, change it to `app.py`.

## 5. How to use WildSpot

1. Upload one clear leopard or tiger photograph.
2. Review the image preview.
3. Press **Run Wildlife Analysis**.
4. Read the prediction, confidence and class probabilities.
5. If an uncertainty message appears, upload a clearer image.

## Important model limitation

The required task is binary classification. The model knows only leopard and
tiger, so an unsupported image can still receive one of those labels with high
confidence. The uncertainty threshold helps only when the highest class score
is low. A future production version should first use a separate validator to
check whether a supported animal is present.

get324-group19-leopard-tiger


## Authors

This project was developed by **GET 324 Group 19** (Mechanical Engineering, University of Uyo) as part of Laboratory Exercise 10 on Machine Learning and Cloud Deployment.

See `CONTRIBUTORS.md` for the complete list of team members and their GitHub usernames.