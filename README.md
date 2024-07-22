# RSPGame_Webcam

### Introduction
<img width="420" alt="image" src="https://github.com/user-attachments/assets/ff4d56e2-fd06-475b-9386-6903bc9b9b48">

In this project, we address the issue of recognizing various versions of scissors in a real-time Rock-Paper-Scissors game using a webcam. This problem is significant because traditional models often fail to generalize well to different or unseen versions of objects within a class. Our goal is to enhance model accuracy and robustness by expanding the training dataset with diverse representations of scissors. We achieved this by including additional images of various scissor types, which improved the model's ability to recognize them accurately in a dynamic setting.

### Methods - Flow
<img width="470" alt="image" src="https://github.com/user-attachments/assets/bbf98bad-d3f7-4bb6-81a0-b0ee760928b4">

1. **Data Collection**: Collect hand gesture data using a webcam.
2. **Feature Extraction**: Extract features based on the landmark data, calculating angles and vectors between joints.
3. **Model Training**: Train the model using the extracted features.
4. **Prediction**: Use the trained model to predict the input hand gestures.
5. **Determine Winner**: Determine the winner of the Rock-Paper-Scissors game based on the predicted results.

#### Hyperparameter Tuning
* Cross-validation techniques were used to find the optimal k value for KNN.
<img width="236" alt="image" src="https://github.com/user-attachments/assets/5b55a79c-2baf-4da8-85d1-bf9ffacf917f">
👉 I performed 5-fold cross-validation for k values ranging from 1 to 20. And I trained the KNN model for each k value and calculated the mean accuracy across the folds.

📌 The cross-validation results showed that k=1 had the highest accuracy. However, to prevent overfitting, we chose k=5, which provided a good balance between accuracy and generalization.

### Result
<img width="423" alt="image" src="https://github.com/user-attachments/assets/79b84eb8-1a60-4300-a7e6-2efd6d5fdd05">

### Reference Code: [github](https://github.com/kairess/Rock-Paper-Scissors-Machine)

---

# How to use code

## Requirements
Make sure you have the following dependencies installed:
- Python 3.x
- OpenCV
- NumPy
- Scikit-learn, Matplotlib, Pandas
- mediapipe

## Step-by-Step Guide

1. **Check Camera**
   Run `check_camera.py` to ensure your webcam is working correctly.
   ```sh
   python check_camera.py
   ```

2. **Create Dataset**
   Running this script will turn on the camera. The index specified by the user will be assigned as a label, and the finger angles will be recorded and saved with a filename in the format raw_{action}_{created_time}.csv. After that, copy the contents from raw_{action}_{created_time}.csv and paste them at the bottom of dataset/gesture_train.csv.
   ```sh
   python create_dataset.py
   ```

3. **Find Optimal k**
   Execute `findOptimalK.py` to determine the optimal k value for the KNN algorithm using cross-validation.
   ```sh
   python findOptimalK.py
   ```

4. **Run Main Application**
   Finally, run `main.py` to start the main application.
   ```sh
   python main.py
   ```
