#!/bin/bash

find . -name "*.$1" | xargs cat | wc
