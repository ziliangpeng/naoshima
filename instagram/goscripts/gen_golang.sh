#!/bin/bash

protoc --go_out=plugins=grpc:. ig.proto
