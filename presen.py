import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MobileNetV2 model with pre-trained weights
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Define the image paths
img_paths = ['meow.png', 'dog.png', 'bird.jpg', 'umbrella.jpg', 'madhu.jpg']

# Define a function to predict and display the image
def predict_and_display(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    input_image = tf.keras.preprocessing.image.img_to_array(img)
    input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)
    input_image = tf.expand_dims(input_image, axis=0)

    # Make predictions
    predictions = model.predict(input_image)
    predicted_classes = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

    # Display the image
    plt.imshow(img, interpolation='bicubic')
    plt.axis('off')
    plt.show()

    # Print the predictions
    print("Predictions:")
    for i, prediction in enumerate(predicted_classes):
        _, class_name, confidence = prediction
        print(f"{i+1}. {class_name}: {confidence * 100:.2f}%")

# Call the function for each image path
for image_path in img_paths:
    predict_and_display(image_path)