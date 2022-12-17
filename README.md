# Food101 - Images classification

Classifying food images is required in a variety of applications. For example, a food images classification model could be used in **a restaurant or grocery store to automatically categorize and organize their food items.** This could make it easier for customers to search for and find specific items, or for the store to keep track of inventory.

Another potential use case for a food images classification model is in a **recipe recommendation system**. By accurately categorizing the ingredients in a food image, the model could help users find recipes that use those ingredients, or suggest alternative ingredients that could be used in the same recipe.

In general, a food images classification model could be useful in any situation where it is necessary to quickly and accurately identify the types of food present in an image. This could include applications in **food safety, nutrition, or even in personal health tracking,** where users may want to automatically track their food intake by taking photos of their meals.

## Dataset 
Food101 images dataset is available at - https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz

- This dataset includes `101` categories of food images - `1000` images per class. 
- We will be using subset of this large dataset - 8 categories with 1000 images per class.
- Classes used: `['cup_cakes', 'french_fries', 'hamburger', 'pizza', 'ramen', 'onion_rings', 'samosa', 'waffles']`

## Model

To solve a **food images classification** problem using a pre-trained model for transfer learning with Keras, we can use the `Xception` model. `Xception` is a deep convolutional neural network that has been trained on the `ImageNet` dataset. By using a pre-trained model for transfer learning, we can leverage the knowledge learned by the model on the `ImageNet` dataset and apply it to our food images classification problem.

To begin, we will need to download the `Xception` model **weights and architecture** from the Keras library. We can then load the model weights and architecture into our Keras model, and add our own fully connected layers on top of the Xception model. These fully connected layers will be trained to recognize the specific classes of food images in our dataset.

Next, we will need to compile our Keras model and specify the loss function and optimizer. We can then use the fit() function to train our model on our food images dataset. As the model trains, it will use the pre-trained Xception weights as a starting point, and update the weights of the fully connected layers to better recognize the classes of food in our dataset.

Once our model has been trained, we can use it to make predictions on new images of food. The Xception model, with its pre-trained weights and added fully connected layers, should be able to accurately classify the classes of food in these new images.

- Imagenet dataset: https://www.image-net.org/
- Pre-trained models: https://keras.io/api/applications/

Afte this, trained model can be converted to TF-Lite format, containerized and deployed to serverless cloud infra for inference. 

## Deployment to cloud - AWS lambda
- For testing, this model is currently deployed to - https://m1i66sda82.execute-api.us-west-2.amazonaws.com/test/predict 
- You can test it from local system using 
```
python test-aws.py
```
- To avoid recurring chagres, this cloud deployment will be teared down in 2 weeks.

## Files included in this repo
1. `README.md` - readme file with description of the problem and instructions on how to run the project
2. `requirements.txt` - dependencies that can be installed with `pip`
3. `notebook.ipynb` - dataset download, data cleaning, preprocessing, EDA, model selection and parameter tuning. This file also includes final model training, saving, loading and inference(prediction) testing
4. `train.py` - training and saving final model
5. `test-tf-model.py` - testing TF model
6. `tflite-notebook.ipynb` - Convert model to TF-Lite format, remove Tensorflow dependency and test prediction
7. `convert.py` - to convert TF model to TF-Lite model
8. `lambda_function.py` - loading the model for inference(prediction) and lanbda handler. The Lambda function handler is the method in Python code that processes events. When a function is invoked, Lambda runs the handler method.
9. `Dockerfile` - a text file of instructions which are used to automate installation and configuration of a Docker image
10. `test-local.py` - to test docker image locally
11. `test-aws.py` - to test AWS lambda/API Gateway from local system

## How to run this project?

**Prerequisites:**
- Docker - installed locally, using https://docs.docker.com/
- System with GPU for experimenting/training the model
- `anaconda` or `miniconda` with `conda` package manager
- Dataset downloaded, extracted and subset (sample) created in `dataset_mini` folder for below 8 classes. If needed, refer to `notebook.ipynb` section 1.1 Obtaining and exploring image data
```
['cup_cakes', 'french_fries', 'hamburger', 'pizza', 'ramen', 'onion_rings', 'samosa', 'waffles']
```
### **Steps**

### A) Setup local environment

1. Create a new conda environment and activate
```
conda create --name food101 python=3.9
conda activate food101
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Clone the repo
```
git clone https://github.com/ranga4all1/food101-classification.git
```

### B) Model training and conversion

1. Run `train.py'. This would save a few model files.
```
python train.py
```
2. Rename saved model file with highest version/score to `food101-model.h5`
```
mv <your-model-file> food101-model.h5
```
3. Test TF model by running below command
```
python test-tf-model.py
```
Output should look like this:
```
{
 'cup_cakes': -1.6670524,
 'french_fries': -1.0787222,
 'hamburger': -1.6100546, 
 'pizza': 0.68576145, 
 'ramen': -0.7817082, 
 'onion_rings': -3.3420057, 
 'samosa': 10.320705, 
 'waffles': 0.4640668
 }
```
4. Run `convert.py'. This would convert TF model to TF-Lite model and test it
```
python convert.py
```
This should generate `food101-model.tflite` file and test this tflite model with test image (samosa.jpg) hosted on Github generating test output similar to this:
```
{
 'cup_cakes': -1.6670524,
 'french_fries': -1.0787222,
 'hamburger': -1.6100546, 
 'pizza': 0.68576145, 
 'ramen': -0.7817082, 
 'onion_rings': -3.3420057, 
 'samosa': 10.320705, 
 'waffles': 0.4640668
 }
```

### C) Containerization

1. Build Docker image
```
docker build -t food101-model .
```
2. Verify docker container created
```
docker images
```
3. Run the docker image locally
```
docker run -it --rm -p 8080:8080 food101-model:latest
```
4. Test docker sandbox
```
python test-local.py
```
Output should look like this:
```
{
 'cup_cakes': -1.6670524,
 'french_fries': -1.0787222,
 'hamburger': -1.6100546, 
 'pizza': 0.68576145, 
 'ramen': -0.7817082, 
 'onion_rings': -3.3420057, 
 'samosa': 10.320705, 
 'waffles': 0.4640668
 }
```
----------------------

### D) Deploy to cloud - AWS lambda + API GAteway

**AWS Lambda**

Note: Collect output of below commands - you would need it for next steps

1. Create ecr repository
```
aws ecr create-repository --repository-name food101-tflite-images
```
2. Get ecr docker login
```
aws ecr get-login --no-include-email | sed 's/[0-9a-zA=]\{20,\}/PASSWORD/g'
```
3. login to ecr
```
$(aws ecr get-login --no-include-email)
```
4. Once logged in, run below commands with your account details captured earlier
```
ACCOUNT=<your-account-number>
REGION=<your-aws-region>
REGISTRY=food101-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
TAG=food101-model-xception-v4-001
REMOTE_URI=${PREFIX}:${TAG}
```

5. verify
```
echo ${REMOTE_URI}
```

6. Tag and Push docker image to remote ecr URI
```
docker images
docker tag food101-model:latest ${REMOTE_URI}
docker images
docker push ${REMOTE_URI}
```
7. Login to AWS console from browser:
    1) Check container registry - pushed docker image should show up in ecr repository
    2) Go to Lambda -> create new function -> Container image
    function name: food101-classification
    3) click -> Browse images -> select repository `food101-tflite-images`-> select image `food101-model-xception-v4-001`

8. Test
    1) Go to `Configuration` tab in food101-classification function. Click EDIT and Change these 2 parameters - 
    ```
    timeout : 30 seconds
    Memory : 1024MB
    ```
    2) Go to `Test` tab in food101-classification function
    3) In 'test event' modify EVENT JSON as below
    ```
    {
       "url": "https://bit.ly/3PzCqJ2"
    }
    ```
    4) click -> Test button
    
    result should show as below:

    Execution result: succeeded(logs)

    Click -> Details
    ```
    The area below shows the last 4 KB of the execution log.
   {
    'cup_cakes': -1.6670524,
    'french_fries': -1.0787222,
    'hamburger': -1.6100546, 
    'pizza': 0.68576145, 
    'ramen': -0.7817082, 
    'onion_rings': -3.3420057, 
    'samosa': 10.320705, 
    'waffles': 0.4640668
    }
    ```
----------------------
**API GAteway**

1) In AWS console, Go to (or search) API Gateway -> click 'create API'
2) Under REST API -> click 'Build'
3) Select REST, New API | API Name - food101-classification | click 'Create API'
4) In Resources - Action -> create new resource
  Resource Name: predict -> click 'Create Resource'
5) In Resources - Action -> create new Method -> POST (click on checkmark) and use below parameters, then click 'Save'
  ```
  Integration type: Lambda Function
  Lambda Function: food101-classification
  ```
6) Select OK in this window- Add Permission to Lambda Function
7) Click on 'Test'; Use this in Request Body: 
```
{"url": "https://bit.ly/3PzCqJ2"}
```
and then click 'Test'

8) Result should look like this:
    ```
    Request: /predict
    Status: 200
    Latency: 2549 ms
    Response Body
    {
 'cup_cakes': -1.6670524,
 'french_fries': -1.0787222,
 'hamburger': -1.6100546, 
 'pizza': 0.68576145, 
 'ramen': -0.7817082, 
 'onion_rings': -3.3420057, 
 'samosa': 10.320705, 
 'waffles': 0.4640668
 }
    ```
9) In Resources - Action -> click Deploy API - Deployment stage: New Stage, Stage name: Test
10) In test Stage Editor -> Copy 'Invoke URL' and then save changes


### E) Test Cloud deployment - API gateway+lambda function

1) On local system, update `test-aws.py` file with `Invoke URL`. Add `/predict` in the end of url.
2) On local system, run below command  
```
python test-aws.py
```
Result :
```
{
 'cup_cakes': -1.6670524,
 'french_fries': -1.0787222,
 'hamburger': -1.6100546, 
 'pizza': 0.68576145, 
 'ramen': -0.7817082, 
 'onion_rings': -3.3420057, 
 'samosa': 10.320705, 
 'waffles': 0.4640668
 }
```

## Notes
- Alternatively this model can be deployed as an API endpoint on any other cloud. Also various deployments tools can be used for those deployments. Please feel free to explore more.
- CAUTION: Deploying model to clould services may incurr charges