#!/bin/zsh

# Run, without leaving a executable file
echo "Running with 'go run'.."
go run main.go

echo ''

# Build executable
rm first-program
echo "Building with 'go build'.."
echo "running ls.."
ls
echo "Building..."
go build -v .
echo "running ls.."
ls
./first-program
