<!--
Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.

Downloading, reproducing, distributing or otherwise using the SDK Software
is subject to the terms and conditions of the Boston Dynamics Software
Development Kit License (20191101-BDSDK-SL).
-->

# Post Docking Callback Examples

The scripts in this folder allow you to upload files saved to the Data Acquisition Store during robot operation to various endpoints, with the target use case having the callback run when Spot docks at the end of an Autowalk mission.

## Install Packages

Run the below to install the necessary dependencies:

```
python3 -m pip install -r requirements.txt
```

## Configuration Requirements

For AWS, you must have your config file saved at `~/.aws/config` with format:

```
[default]
aws_access_key_id=KEY
aws_secret_access_key=KEY
```

If running on a CORE I/O, you will need access to the internet or your local network.

## Run a Callback

Run the scripts by the following commands:
AWS:

```
python3 -m daq_upload_docking_callback --destination aws --bucket-name YOUR_BUCKET --host-ip HOST_COMPUTER_IP SPOT_IP
```

Note: You can either use a config file at `~/.aws/config` or use the `--aws-access-key` and `--aws-secret-key` arguments to have this service create the file.

GCP:

```
python3 -m daq_upload_docking_callback --destination gcp --key-filepath PATH_TO_KEY_JSON --bucket-name YOUR_BUCKET --host-ip HOST_COMPUTER_IP SPOT_IP
```

Local:

```
python3 -m daq_upload_docking_callback --destination local --destination-folder DESTINATION --host-ip HOST_COMPUTER_IP SPOT_IP
```

You can use the optional `--time-period` argument to adjust how far back the callback should look for files. If not specified, the callback will look for files starting from when the callback was initialized. After running once, the start time will be reset.

## Run a Callback using Docker

Please refer to this [document](../../../docs/payload/docker_containers.md) for general instructions on how to run software applications on computation payloads as docker containers.

You can find general instructions on how to build and use the docker image [here](../../../docs/payload/docker_containers.md#build-docker-images).

To build the Docker image, run:

```
sudo docker build -f Dockerfile  -t docking_callback .
sudo docker save docking_callback | pigz > docking_callback.tar.gz
```

To run the Docker image on the same computer where it was built, run:

```
sudo docker run -it --network=host docking_callback --time-period 90 --destination local --destination-folder test_folder --host-ip HOST_IP ROBOT_IP
```

where `--time-period 90 --destination local --destination-folder test_folder --host-ip HOST_IP ROBOT_IP` is one possibility of many for the command-line and positional arguments. This command results in the corresponding data (i.e., the last 90 minutes), if any, being retrieved from Data Acquisition Store service and copied to a folder inside of the Docker container, not the host machine. To copy this folder from the Docker container to the host machine, run:

```
sudo docker cp CONTAINER_ID:/app/test_folder .
```

where `CONTAINER_ID` is the container ID for the `docking_callback` image.

To build the Docker image for an arm64 platform (e.g. CORE I/O), run:

```
# Prerequisites
# Install the pigz and qemu packages
sudo apt-get install qemu binfmt-support qemu-user-static pigz
# This step will execute the registering scripts
sudo docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Builds the image
sudo docker build -f Dockerfile.arm64  -t docking_callback:arm64 .
# Exports the image, uses pigz
sudo docker save docking_callback:arm64 | pigz > docking_callback_arm64.tar.gz
```

Note: For the AWS callback, you must copy your config file as `config` to this directory for `docker build` to work. You will then uncomment `COPY ./config ~/.aws/config` in Dockerfile. Alternatively, you can supply your keys by using the `--aws-access-key` and `--aws-secret-key` arguments.

This example can also be built into a [Spot Extension](../../../docs/payload/docker_containers.md) using a provided [convenience script](../extensions/README.md)

```
cd {/path/to/python/examples/post_docking_callbacks/}

python3 ../extensions/build_extension.py \
    --dockerfile-paths Dockerfile.arm64 \
    --build-image-tags docking_callback:arm64 \
    -i aws_docking_callback_arm64.tar.gz \
    --package-dir . \
    --spx aws_docking_callback.spx
```

The output file will be called aws_docking_callback_arm64.spx and can be uploaded to a CORE I/O.
