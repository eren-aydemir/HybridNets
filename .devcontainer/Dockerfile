# Building on top of Bvidia Pytorch NGC Container 
FROM nvcr.io/nvidia/pytorch:22.05-py3

# install pycuda
RUN pip3 install pycuda

# install icontract for contracts
RUN pip3 install icontract

# install determined.ai
RUN pip3 install determined

# install colored version
RUN python3 -m pip install colored

# install graph surgeon for onnx
RUN pip3 install onnx_graphsurgeon

# install onnxruntime
RUN pip3 install onnxruntime

# install HybridNets requirements
RUN pip3 install prefetch_generator>=1.0.1
RUN pip3 install timm>=0.5.4
RUN pip3 install webcolors>=1.11.1
RUN pip3 install pretrainedmodels>=0.7.4
RUN pip3 install efficientnet_pytorch>=0.7.1
RUN pip3 install path
RUN pip3 install tensorboardX
RUN pip3 install albumentations>=1.1.0
RUN pip3 install opencv_python_headless==4.1.2.30
RUN pip3 install seaborn>=0.11.2
RUN pip3 install tqdm>=4.61.2
RUN pip3 install PyYAML>=3.13
RUN pip3 install monai
RUN pip3 install torch-tb-profiler==0.4.0

# Switch to non-root user
RUN useradd -m -d /home/dgxuser -s /bin/bash -u 1001 dgxuser
USER dgxuser
WORKDIR /home/dgxuser