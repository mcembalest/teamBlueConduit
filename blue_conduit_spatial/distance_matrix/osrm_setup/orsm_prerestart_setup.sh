#!/bin/bash

sudo apt update
sudo apt install docker.io
sudo service docker start
sudo usermod -a -G docker ubuntu

# MUST EXIT OUT & BACK INTO INSTANCE via SSH


