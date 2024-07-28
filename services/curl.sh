#!/bin/bash

curl -vv -X POST -H "Content-Type: application/json" -d '{"input_integer": 42}' http://127.0.0.1:4200/echo