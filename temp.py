import tkinter as tk
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tkinter import filedialog, colorchooser

# Load the trained model
model = load_model('mnist_model.keras')

# Set up the tkinter window
window = tk.Tk()
window.title("Digit Recognition")
window.geometry("500x600")

# Set initial theme (Light)
theme = "light"

# Function to apply light theme
def apply_light_theme():
    window.config(bg="white")
    result_label.config(bg="white", fg="black")
    instructions_label.config(bg="white", fg="black")
    canvas.config(bg="white")
    reset_button.config(bg="lightgrey", fg="black")
    theme_toggle_button.config(bg="lightgrey", fg="black")
    menu_bar.config(bg="lightgrey")

# Function to apply dark theme
def apply_dark_theme():
    window.config(bg="black")
    result_label.config(bg="black", fg="white")
    instructions_label.config(bg="black", fg="white")
    canvas.config(bg="black")
    reset_button.config(bg="grey", fg="black")
    theme_toggle_button.config(bg="grey", fg="black")
    menu_bar.config(bg="grey")

# Toggle theme function
def toggle_theme():
    global theme
    if theme == "light":
        apply_dark_theme()
        theme = "dark"
        theme_toggle_button.config(text="Switch to Light Theme")
    else:
        apply_light_theme()
        theme = "light"
        theme_toggle_button.config(text="Switch to Dark Theme")

# Create the main frame for organizing widgets
main_frame = tk.Frame(window)
main_frame.pack(padx=10, pady=10)

# Create a frame for the drawing canvas
canvas_frame = tk.Frame(main_frame)
canvas_frame.grid(row=0, column=0, columnspan=2, pady=10)

# Create a canvas for drawing (scaled)
canvas = tk.Canvas(canvas_frame, width=400, height=400, bg="white")
canvas.pack()

# Initialize a list to store the points of the drawn digit
points = []

# Set initial drawing color
draw_color = "black"

# Function to capture mouse drag events (drawing)
def draw(event):
    x, y = event.x, event.y
    points.append((x, y))
    canvas.create_oval(x-5, y-5, x+5, y+5, fill=draw_color)

# Bind mouse drag to the drawing function
canvas.bind("<B1-Motion>", draw)

# Function to reset the canvas
def reset_canvas():
    canvas.delete("all")
    points.clear()

# Create a frame for the control buttons (Clear, Recognize, Save, etc.)
control_frame = tk.Frame(main_frame)
control_frame.grid(row=1, column=0, columnspan=2, pady=10)

# Button to reset the canvas
reset_button = tk.Button(control_frame, text="Reset", command=reset_canvas)
reset_button.grid(row=0, column=0, padx=5)

# Button to recognize the digit
predict_button = tk.Button(control_frame, text="Recognize", command=lambda: recognize_digit())
predict_button.grid(row=0, column=1, padx=5)

# Button to switch between light and dark themes
theme_toggle_button = tk.Button(control_frame, text="Switch to Dark Theme", command=toggle_theme)
theme_toggle_button.grid(row=1, column=0, columnspan=2, pady=5)

# Label to show the prediction result
result_label = tk.Label(main_frame, text="Predicted digit: None\nConfidence: None")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

# Add a label for instructions
instructions_label = tk.Label(main_frame, text="Draw a digit on the canvas and click 'Recognize' to predict.")
instructions_label.grid(row=3, column=0, columnspan=2, pady=5)

# Function to recognize the digit
def recognize_digit():
    if not points:
        result_label.config(text="Error: Please draw a digit first!")
        return

    # Create an empty image to draw the digit (white background)
    img = Image.new("L", (28, 28), color="white")
    img_draw = ImageDraw.Draw(img)  # Use ImageDraw to draw on the image

    # Normalize the drawing points to fit in the 28x28 image space
    for point in points:
        x, y = point
        # Map the canvas coordinates to the 28x28 image space
        x = int((x / canvas.winfo_width()) * 28)  # Normalizing the x coordinate to 28x28
        y = int((y / canvas.winfo_height()) * 28)  # Normalizing the y coordinate to 28x28
        img_draw.point((x, y), fill="black")

    # Resize and invert image to match MNIST format (white background and black digits)
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    # Convert image to numpy array and preprocess
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict the digit using the model
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Update the result label with the prediction and confidence
    result_label.config(text=f"Predicted digit: {digit}\nConfidence: {confidence*100:.2f}%")

# Function to save the drawing as an image
def save_drawing():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        # Create an empty image to draw the digit
        img = Image.new("L", (28, 28), color="white")
        img_draw = ImageDraw.Draw(img)  # Use ImageDraw to draw on the image

        # Normalize the drawing points
        for point in points:
            x, y = point
            # Map the canvas coordinates to the 28x28 image space
            x = int((x / canvas.winfo_width()) * 28)
            y = int((y / canvas.winfo_height()) * 28)
            img_draw.point((x, y), fill="black")

        # Save the drawing
        img.save(file_path)

# Function to select a color for drawing
def choose_color():
    global draw_color
    color = colorchooser.askcolor()[1]  # Open color chooser and get the hex code
    if color:
        draw_color = color  # Update the drawing color

# Create a menu bar
menu_bar = tk.Menu(window)

# Add a "File" menu with options
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Save Drawing", command=save_drawing)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Add a "Theme" menu to switch between themes
theme_menu = tk.Menu(menu_bar, tearoff=0)
theme_menu.add_command(label="Switch to Dark Theme", command=toggle_theme)
menu_bar.add_cascade(label="Theme", menu=theme_menu)

# Add a "Color" menu to choose drawing color
color_menu = tk.Menu(menu_bar, tearoff=0)
color_menu.add_command(label="Choose Drawing Color", command=choose_color)
menu_bar.add_cascade(label="Color", menu=color_menu)

# Configure the window to use the menu
window.config(menu=menu_bar)

# Apply light theme initially
apply_light_theme()

# Run the tkinter event loop
window.mainloop()
