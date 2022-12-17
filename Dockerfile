FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras-image-helper
RUN pip install tflite_runtime==2.7

COPY food101-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]