import tensorflow as tf
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir,'model.tflite')
interpretor = tf.lite.Interpreter(model_path=model_path)
interpretor.allocate_tensors()
#load our labels / list of classes
labels_path = os.path.join(script_dir, 'labels.txt')
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

#Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize(224,224)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = np.array(img, dtype=np.float32)
    img_array /= 255.0
    return img_array

    #Predict the class of the image and confidence level
    def classify_image(image_path):
        img_array = load_and_preprocess_image(image_path)
        input_details = interpretor.get_input_details()
        output_detail = interpretor.get_output_details()

        interpretor.set_tensor(input_detail[0]['index'], img_array)
        interpretor.invoke()
        predictions = interpretor.get_tensor(output_detail[0]['index'])
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)
        return labels[predicted_class[0]], confidence
if __name__ == "__main__":
    for file_name in os.listdir(script_dir):
        if file_name.lower().endswith(('.jpg','jpeg','.png')):
            image+path = os.path.join(script_dir, file_name)
            predicted_class, confidence = classify_image(image_path)
            print(f'Image: {file_name} | Predicted class: {predicted_class} | Confidence: {confidence: .2f}')
