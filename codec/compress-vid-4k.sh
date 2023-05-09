#!/bin/bash

avconvert --preset PresetHEVC3840x2160 --disableMetadataFilter --source $1 --output $1.hevc4k.MOV --progress

