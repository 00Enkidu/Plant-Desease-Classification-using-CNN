# Plant Disease Classification using CNN

A deep learning project for automated plant disease detection using Convolutional Neural Networks (CNNs). Includes model training, evaluation, an interactive Streamlit web app for real-time image diagnosis, and Docker support for easy deployment. Achieved high accuracy on leaf image datasets through data augmentation and robust model design.

---

## About the Dataset

To meet the growing food demand—projected to require a 70% increase in production by 2050—addressing crop diseases is crucial. Infectious diseases currently reduce crop yields by an average of 40%, with some farmers suffering total losses. However, the widespread use of smartphones in agriculture (expected to reach 5 billion devices by 2020) creates new opportunities for rapid disease diagnostics via machine learning and mobile technology.

This project utilizes the PlantVillage dataset, which contains over 50,000 expertly curated images of healthy and diseased plant leaves. This dataset supports the development of computer vision tools to help mitigate yield losses caused by crop diseases.

- **Dataset:**  PlantVillage  
- **Number of Classes:** 38 (covers various crops and diseases)
- **Images:** 50,000+ high-quality images of healthy and infected plant leaves  
- **Source:** [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

*These data are part of an ongoing, crowdsourced effort to empower growers and researchers with intelligent disease detection tools.*

---

## Image Processing

Image preprocessing is a crucial step for ensuring consistent input and improving model accuracy. This project’s image processing pipeline is implemented primarily in `app/main.py` and involves the following steps:

```python
def load_and_perprocess_image_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array
```

**Key Steps:**
- **Resizing:** All input images are resized to the specified `target_size` (default: 224x224 pixels) to ensure consistent input size.
- **Normalization:** Pixel values are scaled to the [0, 1] range to stabilize and speed up model training.
- **Dimension Expansion:** The image array is expanded to add a batch dimension, matching the model's expected input shape `[batch, height, width, channels]`.

> **Note:**  
> - The function operates on RGB color images without grayscale conversion.
> - The `target_size` should match the input shape expected by the model.
> - Data augmentation (e.g., rotation, flipping) is not included in this function but may be used during model training in the Jupyter Notebook.

---

## Model Training and Architecture

Model training is implemented in the Jupyter Notebook (`Model Notebook/Plant_Disease_Prediction_with_CNN.ipynb`). The main model architecture is as follows:

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))
```

**Model Structure:**
- **Input Layer:** Accepts color images of shape `(img_size, img_size, 3)`.
- **Convolutional Layers:** Extract features from the input images.
- **Pooling Layers:** Reduce the spatial dimensionality and computation.
- **Flatten Layer:** Converts the 2D feature maps to a 1D vector.
- **Dense Layers:** Fully connected layers for high-level reasoning.
- **Output Layer:** Softmax activation for multi-class classification (number of classes equals `train_generator.num_classes`).

**Training Details:**
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** `adam`
- **Metrics:** `accuracy`

**Results**
- The model is trained for 5 epochs, achieving remarkable performance:
  - **Training Accuracy:** Improved from 60.2% to 98.6%
  - **Training Loss:** Decreased from 1.63 to 0.044
  - **Validation Accuracy:** Remained above 87% throughout training

```
Epoch 1/5
1358/1358 ━━━━━━━━━━━━━━━━━━━━ 102s 70ms/step - accuracy: 0.6015 - loss: 1.6271 - val_accuracy: 0.8744 - val_loss: 0.4038
Epoch 2/5
1358/1358 ━━━━━━━━━━━━━━━━━━━━ 129s 65ms/step - accuracy: 0.9221 - loss: 0.2475 - val_accuracy: 0.8715 - val_loss: 0.3935
Epoch 3/5
1358/1358 ━━━━━━━━━━━━━━━━━━━━ 142s 65ms/step - accuracy: 0.9657 - loss: 0.1099 - val_accuracy: 0.8926 - val_loss: 0.3904
Epoch 4/5
1358/1358 ━━━━━━━━━━━━━━━━━━━━ 91s 67ms/step - accuracy: 0.9783 - loss: 0.0668 - val_accuracy: 0.8838 - val_loss: 0.5069
Epoch 5/5
1358/1358 ━━━━━━━━━━━━━━━━━━━━ 139s 65ms/step - accuracy: 0.9858 - loss: 0.0441 - val_accuracy: 0.8827 - val_loss: 0.5362

Evaluating model
339/339 ━━━━━━━━━━━━━━━━━━━━ 17s 50ms/step - accuracy: 0.8824 - loss: 0.5384
Validation Accuracy: 88.27%
```

The final trained model is saved in `.h5` format and used for prediction in the Streamlit web app.

---
## Result Visualization
Below is the plot of training and validation accuracy and loss over epochs:
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/630be693-ac32-4696-832a-c43da20cf5f8" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/91f84002-5148-4196-878b-806a07d75c9f" />

---

## Model Training Summary
The training results (see accuracy and loss plots above) show that the model achieves high training accuracy and significant reduction in loss over epochs, indicating effective learning on the training set. The validation accuracy remains stable at a high level (around 88–89%), although there is a noticeable gap between training and validation accuracy. This suggests that while the model generalizes well, there may be some overfitting, and further improvements such as regularization or data augmentation could be considered. Overall, the model demonstrates strong potential for practical plant disease classification tasks.

---

## Streamlit Web Application

- The project includes an interactive [Streamlit](https://streamlit.io/) app (`main.py`) for user-friendly, real-time disease diagnosis.
- **How to use:**
  1. Run `main.py` locally or via Docker (see below).
  2. Upload an image of a plant leaf.
  3. Instantly receive the predicted disease class and confidence score.

---

## Docker Deployment

- The project supports containerized deployment for easy setup and consistent environments.
- **How to use:**
  1. Build the Docker image:
     ```bash
     docker build -t plant-disease-classifier .
     ```
  2. Run the Docker container:
     ```bash
     docker run -p 8501:8501 plant-disease-classifier
     ```
  3. Access the Streamlit app at: [http://localhost:8501](http://localhost:8501)

---

## Repository Structure

- `main.py` — Streamlit app for plant disease classification
- `Dockerfile` — Instructions for containerizing and running the application
- `notebooks/` — Jupyter Notebooks for data exploration, model development, and training logs
- `models/` — Saved model weights and architecture (if available)
- `requirements.txt` — Python dependencies

---

## References

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

---

## Acknowledgements

This project is inspired by the need for scalable, accessible technology to support global food security and empower farmers through AI-driven disease detection.

---

> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**
