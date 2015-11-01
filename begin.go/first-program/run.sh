#!/bin/zsh

# Run, without leaving an executable file
echo " >>> Running with 'go run'.. This will not leave an executable"
go run main.go
echo ''

# Build executable
if [ -f 'first-program' ];
then
    echo ' >>> `first-program` exist. deleting..'
    rm first-program
fi
echo " >>> To build with 'go build'.."
echo " >>> running ls.."
ls
echo " >>> Building..."
go build -v .
echo " >>> running ls.."
ls
./first-program
rm first-program
