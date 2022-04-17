# Identifying Cracks on Concrete with Image Classification using Convolutional Neural Network

## 1. Summary
The project is to create a model using convolutional neural network (CNN) that is able to differentiate concrete with and without cracks. The dataset consist of images with 20000 positive cracks and 20000 negative cracks image. THe dataset is can be obatained on https://data.mendeley.com/datasets/5y9wdsg2zt/2

## 2. IDE and Framework
IDE - Spyder
Frameworks - TensorFlow Keras, Numpy & Matplotlib

## 3. Methodology
### 3.1 Data Pipeline
The image is loaded from the local path and will be split into train:validation:test which is 70:24:6 split ratio.

### 3.2 Model Pipeline
The input layer image dimension is set to 160x160 which represents in (160,160,3) in 3 dimensionsal shape.
Transfer learning is applied in the model.
Prefetch dataset is created to give the model better peformance.
Data augmentation is applied in the model.
mobilenet_v2 preprocessing is used to rescale the input and will be frozen so it will not be update during model training.
Global average pooling and dense layer is used to classify output, and softmax is used to identify predicted class
The model structure is shown as in figure below
![model structure](https://user-images.githubusercontent.com/100821053/163702854-1f2fcb24-7648-422c-8ea9-5c855b8e50bc.png)

The model is trained with batch size of 512 and 10 epochs. The training model accuracy is 99% and validation accuracy is also 99%. The training process is illustrated as in figure below:
![loss](https://user-images.githubusercontent.com/100821053/163702922-75e0c257-06a5-4c1a-bd99-dc596fa1fdd2.png)
![accuracy](https://user-images.githubusercontent.com/100821053/163702925-82e06cfa-edc4-47ef-b2ff-ee100af96e37.png)

## 4. Results

The test data result is shown in figure below
![Test result](https://user-images.githubusercontent.com/100821053/163702932-ac64fb2f-d796-4069-8254-e03cb977021a.png)

Predictions is performed with the test data and the result obtained is as the figure shown
![prediction](https://user-images.githubusercontent.com/100821053/163702958-6bc16ff4-8962-4191-8976-dce9d19569df.png)
Overall the model shows great accuracy and predictions which able to differentiate image with and without cracks.
