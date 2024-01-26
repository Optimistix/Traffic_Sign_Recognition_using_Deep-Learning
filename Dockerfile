FROM public.ecr.aws/lambda/python:3.10

RUN pip install pillow
RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY predict.py .
COPY convnet_from_scratch_with_a_dropout_layer.keras.tflite .

CMD [ "predict.lambda_handler"]
