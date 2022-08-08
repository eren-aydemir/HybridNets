# Building on top of Bvidia Pytorch NGC Container 
FROM nvcr.io/nvidia/pytorch:22.05-py3

# install dependencies
RUN pip3 install pycuda && \
    pip3 install icontract && \
    pip3 install determined && \
    pip3 install colored && \
    pip3 install onnx_graphsurgeon && \
    pip3 install onnxruntime && \
    pip3 install prefetch_generator>=1.0.1 && \
    pip3 install timm>=0.5.4 && \
    pip3 install webcolors>=1.11.1 && \
    pip3 install pretrainedmodels>=0.7.4 && \
    pip3 install efficientnet_pytorch>=0.7.1 && \
    pip3 install path && \
    pip3 install tensorboardX && \
    pip3 install albumentations>=1.1.0 && \
    pip3 install opencv_python_headless==4.1.2.30 && \
    pip3 install seaborn>=0.11.2 && \
    pip3 install tqdm>=4.61.2 && \
    pip3 install PyYAML>=3.13 && \
    pip3 install pytorch-lightning && \
    pip3 install monai && \
    pip3 install torch-tb-profiler==0.4.0

# Switch to non-root user
RUN useradd -m -d /home/dgxuser -s /bin/bash -u 1001 dgxuser
USER dgxuser
WORKDIR /home/dgxuser