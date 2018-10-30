FROM tensorflow/tensorflow:1.11.0-py3

ARG model_dir
ENV MODEL_DIR=$model_dir

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp.py ./
COPY $model_dir ./$model_dir
COPY model_runtime ./model_runtime

ENV WEB_APP_PORT=5000
EXPOSE $WEB_APP_PORT

ENTRYPOINT ["python", "webapp.py"]