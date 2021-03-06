#!/bin/bash
# Build YoloV5 TensorRT Plugins
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes make cmake
mkdir -p `dirname $0`/yolov5/build
cd $_ && cmake .. && make
mkdir -p /opt/tritonserver/trt_plugins
mv libyoloplugin.so $_
mv yolov5 $_
# Clean up
#apt-get remove --purge --yes git cmake
