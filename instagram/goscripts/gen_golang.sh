#!/bin/bash

protoc --go_out=plugins=grpc:ig ig.proto
