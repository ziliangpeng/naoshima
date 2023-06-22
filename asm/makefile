test: test.o
	ld -o test test.o -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64 

test.o: test.s
	as -arch arm64 -o test.o test.s