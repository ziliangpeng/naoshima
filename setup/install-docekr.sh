#!/bin/bash

curl -fsSL get.docker.com | bash


sudo usermod -aG docker $USER
