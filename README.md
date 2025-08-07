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

- Images are pre-processed using resizing, normalization, and data augmentation (including rotation, flipping, and zooming) to improve model robustness.
- Data is split into training, validation, and testing sets to ensure fair and effective evaluation.

---

## Model Training and Results

- A Convolutional Neural Network (CNN) is designed and trained for multi-class plant disease classification.
- The model is trained for 5 epochs, achieving remarkable performance:
  - **Training Accuracy:** Improved from 60.2% to 98.6%
  - **Training Loss:** Decreased from 1.63 to 0.044
  - **Validation Accuracy:** Remained above 87% throughout training

```
Epoch 1: accuracy: 0.6015, loss: 1.6271, val_accuracy: 0.8744, val_loss: 0.4038
Epoch 2: accuracy: 0.9221, loss: 0.2475, val_accuracy: 0.8715, val_loss: 0.3935
Epoch 3: accuracy: 0.9657, loss: 0.1099, val_accuracy: 0.8926, val_loss: 0.3904
Epoch 4: accuracy: 0.9783, loss: 0.0668, val_accuracy: 0.8838, val_loss: 0.5069
Epoch 5: accuracy: 0.9858, loss: 0.0441, val_accuracy: 0.8827, val_loss: 0.5362
```

- The final model demonstrates strong generalization and can accurately classify unseen plant leaf images.

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
