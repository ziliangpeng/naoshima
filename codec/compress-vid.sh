#!/bin/bash

avconvert --preset PresetHEVC1920x1080 --disableMetadataFilter --source $1 --output $1.hevc1080.MOV --progress

