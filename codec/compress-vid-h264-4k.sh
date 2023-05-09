#!/bin/bash

avconvert --preset Preset3840x2160 --disableMetadataFilter --source $1 --output $1.h264.4k.MOV --progress

