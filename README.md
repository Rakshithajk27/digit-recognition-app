---

# Handwritten Digit Recognition with Tkinter and MNIST 🎨🤖

Welcome to the Handwritten Digit Recognition Project! This interactive application combines the power of machine learning and Python's Tkinter GUI library to recognize handwritten digits drawn on a canvas. Dive into the world of AI with this fun and educational project! 🚀

## 🌟 Features

- **Interactive Canvas**: Draw digits using your mouse on the Tkinter canvas.
- **Digit Recognition**: Predicts the digit and displays the confidence level.
- **Theme Support**: Toggle between light and dark modes for better aesthetics.
- **Save Your Work**: Save your drawings as images.
- **Color Picker**: Choose your preferred drawing color.
- **User-Friendly Menu**: Includes options to save, toggle themes, and select colors.

## 🛠️ Technologies Used

- **Python**
- **Tkinter**: For GUI development.
- **PIL (Pillow)**: For image processing.
- **NumPy**: For numerical operations.
- **Machine Learning**
  - **TensorFlow/Keras**: For training and loading the MNIST digit recognition model.

## 🚀 Installation and Setup

Follow these steps to get the application up and running:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rakshithajk27/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy pillow
   ```

3. **Download the Pretrained Model**:
   - Place the `mnist_model.keras` file in the project directory.

4. **Run the Application**:
   ```bash
   python temp.py
   ```

## 📂 Project Structure

```
handwritten-digit-recognition/
├── temp.py                # Main application file
├── mnist_model.keras      # Pretrained MNIST model
├── LICENSE                # License information
├── README.md              # Project documentation
└── .gitignore             # Git ignored files
```

## 🖌️ How to Use

1. **Draw**: Use your mouse to draw a digit on the canvas.
2. **Recognize**: Click the **Recognize** button to see the predicted digit and its confidence.
3. **Reset**: Clear the canvas using the **Reset** button.
4. **Save**: Save your masterpiece using the **File > Save Drawing** option.
5. **Change Color**: Choose your preferred drawing color under the **Color** menu.
6. **Toggle Theme**: Switch between light and dark modes with the **Theme** menu or the toggle button.

## 📜 License

This project is licensed under the [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/). You are free to:

- Copy, modify, distribute, and perform the work, even for commercial purposes, without asking permission.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 💬 Feedback

If you have any feedback or suggestions, please reach out by creating an issue or contacting me at rakshithajk27@gmail.com .

---
