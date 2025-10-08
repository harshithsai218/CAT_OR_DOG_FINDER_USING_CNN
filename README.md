# 🐾 Cat vs Dog Image Classifier  

🔗 **Try it here:** [cat-dog-finder](https://cat-dog-finder.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.x-blue)  
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-yellow)  
![Status](https://img.shields.io/badge/Status-Completed-success)  

---

## 📌 Project Overview  
This project classifies images as **Cats or Dogs** using a **Convolutional Neural Network (CNN)** built with **TensorFlow & Keras**.  
It can predict whether a random image (from your folder or via URL) is a **cat 🐱** or a **dog 🐶**.  

---

## 🎯 Objectives  
- Build and train a **CNN model** to distinguish between cats and dogs.  
- Use **image augmentation** for better generalization.  
- Validate model performance using a separate **test dataset**.  
- Predict images from **local directories or URLs** dynamically.  

---

## 🛠️ Technologies Used  
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, PIL, Requests  
- **Dataset:** Custom folders (`training_set`, `test_set`)  
- **Environment:** Jupyter Notebook / Colab / Anaconda  

---

## 📊 Workflow  

### 1. **Data Preparation**  
- Organize dataset into:
```
training_set/
├── cats/
└── dogs/
test_set_1/
├── cats/
└── dogs/
```
- Images are resized to **100x100** pixels.  
- Training data is **augmented** (shear, zoom, horizontal flip).  

### 2. **Model Architecture**
A simple CNN model built using Keras Sequential API:
```python
model = Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
  MaxPooling2D((2,2)),
  Conv2D(32, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(1, activation='sigmoid')
])
```
Layers Breakdown:
🧠 Conv2D: Extracts spatial features (edges, textures).
🌀 MaxPooling2D: Reduces feature map size.
🧩 Flatten: Converts 2D maps into 1D vectors.
⚙️ Dense Layers: Performs classification (binary).

⚙️ Training the Model
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, epochs=50, validation_data=test_generator)
```
Loss: Binary Crossentropy (since it's a 2-class problem).
Optimizer: Adam (adaptive gradient optimization).
Epochs: 50 for better convergence.

🔍 Model Evaluation
```python
model.evaluate(test_generator)
```
Outputs:
Accuracy on test images.
Validation loss trend can be visualized with matplotlib.

🧠 Prediction from Local Folder
Randomly picks an image and predicts:
```python
image_url = 'https://t4.ftcdn.net/jpg/...jpg'
img_array = process_image(image_url, img_width, img_height)
pred = model.predict(img_array)
```
If valid, it downloads → preprocesses → displays → predicts.

📈 Results
Achieved high accuracy on both training and test datasets.
Model generalizes well to unseen web images.
Provides a real-time visual prediction for any input image.

🚀 How to Run the Project
1. Clone the Repository
```
git clone https://github.com/harshithsai218/Cat-Dog-Finder.git
cd Cat-Dog-Finder
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Run the Python Script
```
python app.py
```
📁 Project Structure
```
Cat-Dog-Finder/
│
├── training_set/
│   ├── cats/
│   └── dogs/
│
├── test_set_1/
│   ├── cats/
│   └── dogs/
│
├── cat_dog_finder.py
├── requirements.txt
├── .gitignore
└── README.md
```
🧾 Important Points (What Happens in Code)
ImageDataGenerator → Augments and normalizes images.
CNN Model → Learns to distinguish cats vs dogs.
Training → Uses batches of 64 images.
Prediction (Local + URL) → Accepts random local or online image.
Output → Displays image + prints predicted label.

🏁 Final Outcome

✅ Model successfully classifies cats and dogs with high accuracy.
🐕‍🦺 Demonstrates Deep Learning and Computer Vision fundamentals.
🖥️ Ready for web app deployment using Streamlit.
