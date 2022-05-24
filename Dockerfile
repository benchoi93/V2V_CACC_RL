FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

COPY . /app
WORKDIR /app

# RUN pip install --upgrade pip

RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt

RUN git submodule init
RUN git submodule update
RUN cd ./baselines
RUN pip install -e .

RUN cd ..

CMD ["tensorboard" , "--logdir=logs" , "--bind_all"]