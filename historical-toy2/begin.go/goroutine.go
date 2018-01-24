package main

import (
	"fmt"
)

func increment(input chan int, output chan<- int) {
	output <- (<-input + 1)
}

func chain() {
	const n = 999
    cs := make([]chan int, n+1)
	for i := 0; i <= n; i++ {
		cs[i] = make(chan int)
	}

	for i := 0; i < n; i++ {
		go increment(cs[i], cs[i+1]) // need `go` keyword to put it to goroutine scheduler
	}
	cs[0] <- 1

	fmt.Println(<-cs[n])
}

func fib(n int, output chan<- int) {
	if n <= 1 {
		output <- 1
	} else {
		c1 := make(chan int)
		c2 := make(chan int)
		go fib(n-1, c1)
		go fib(n-2, c2)
		output <- (<-c1 + <-c2)
	}
}

func cal_fib(n int, barrier chan bool) {
	c := make(chan int)
	go fib(n, c)
	fmt.Println("fib of", n, "is", <-c)
	barrier <- true
}

func main() {
	chain()

	barrier := make(chan bool)
	n := 25
	for i := n; i >= 0; i-- {
		go cal_fib(i, barrier)
	}
	for i := n; i >= 0; i-- {
		<-barrier
	}
}
