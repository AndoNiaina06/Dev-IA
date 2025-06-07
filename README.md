# 🧠 CIFAR-10 Image Classifier with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model achieves image classification across 10 object categories and prints performance metrics at the end.

---

## 📁 Project Structure

.
├── main.py # Main script to train and evaluate the CNN
├── image_classifier.h5 # Saved trained model (after training)
└── README.md # Project documentation

yaml
Copier
Modifier

---

## 🧰 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

Install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
🚀 How to Run
Clone the repository or download main.py.

Run the script:

bash
Copier
Modifier
python main.py
The script will:

Load and normalize the CIFAR-10 dataset.

Display sample training images with their labels.

Build and train a CNN for 10 epochs.

Evaluate the model on the test set and print the loss and accuracy.

Save the trained model as image_classifier.h5.

🖼️ Dataset
CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

Classes:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

🧠 Model Architecture
3 convolutional layers with ReLU activation and max pooling

1 fully-connected hidden layer

1 output layer with softmax for 10-class classification

📊 Example Output
At the end of training, you’ll see evaluation metrics like:

makefile
Copier
Modifier
Loss: 0.85
Accuracy: 0.72
(Actual results may vary depending on training performance)

💾 Model Saving
After training, the model is saved to disk as image_classifier.h5.
You can reload it later for inference or further training.

📌 Notes
Ideal for beginners learning CNNs and image classification.

Visualization of training images helps understand data distribution.

vbnet
Copier
Modifier

Let me know if you want it tailored for GitHub with badges, usage examples, or links!







