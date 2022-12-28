#!/bin/bash

cd alpine-3.16/
docker build -t registry.gitlab.com/muspectre/muspectre:alpine3.16 .
docker push registry.gitlab.com/muspectre/muspectre:alpine3.16
cd -

cd debian-stable/
docker build -t registry.gitlab.com/muspectre/muspectre:debian_stable .
docker push registry.gitlab.com/muspectre/muspectre:debian_stable
cd -

cd debian-testing/
docker build -t registry.gitlab.com/muspectre/muspectre:debian_testing .
docker push registry.gitlab.com/muspectre/muspectre:debian_testing
cd -

# cd ubuntu-16.04/
# docker build -t registry.gitlab.com/muspectre/muspectre:ubuntu16.04 .
# docker push registry.gitlab.com/muspectre/muspectre:ubuntu16.04
# cd -

# cd ubuntu-18.04/
# docker build -t registry.gitlab.com/muspectre/muspectre:ubuntu18.04 .
# docker push registry.gitlab.com/muspectre/muspectre:ubuntu18.04
# cd -

# cd ubuntu-20.04/
# docker build -t registry.gitlab.com/muspectre/muspectre:ubuntu20.04 .
# docker push registry.gitlab.com/muspectre/muspectre:ubuntu20.04
# cd -
