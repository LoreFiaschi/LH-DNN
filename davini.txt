docker run -v $(pwd):/cini --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -ti nvcr.io/nvidia/pytorch:24.01-py3
