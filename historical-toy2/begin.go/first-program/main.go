// first line of code must declare package
// package main with func main() will be entry point
package main

import (
    "fmt"
    "time"
)

func greeting() string { // return type after function name
    var now string // declare variable. assigned default value
    now = time.Now().String()
    msg := "Hello world, the time is: " // fast way to declare + assign value
    return msg + now
}

func main() {
    fmt.Println(greeting())
}

