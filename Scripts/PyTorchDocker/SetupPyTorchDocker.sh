# For docker-compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Also, see
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# Check for the latest version on this page, on the left:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Look for the PyTorch Release Notes.
# Also, try
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# and click on "Tags" on the top center for the different tags.

# As of Sept. 1, 2023
docker pull nvcr.io/nvidia/pytorch:23.08-py3

# --rm Removes the container, meant for short-lived process, perform specific task,
# upon exit. You ensure temporary containers don't accumulate on system. --rm only
# removes container instance not the image.
docker run -v /home/propdev/Prop/cuBlackDream:/cuBlackDream --gpus all -it --rm nvcr.io/nvidia/pytorch:23.08-py3

# -p 8888:8888 maps port 8888 inside container to port 8888 on host machine.

# When running this Docker, it said SHMEM (Shared memory?) is set too low, and so it suggested
# these options: --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

docker run -v /home/propdev/Prop/cuBlackDream:/cuBlackDream --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -it -p 8888:8888 --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:23.08-py3

# You may see the following if you run 
jupyter notebook
# in the command prompt:
#     Or copy and paste this URL:
#        http://hostname:8888/?token=cf73dc6455ca527875631fde4f24511067f751e3478c5482
# You want to keep the token. But hostname you may have to replace with localhost, and so this
# works in your browser:
#        http://localhost:8888/?token=cf73dc6455ca527875631fde4f24511067f751e3478c5482
