package main

import (
    "fmt"
    "strconv"
)

func print_map(m map[string]int) {
    // iterating a map
    for k, v := range m {
        fmt.Println(k, v)
    }
}

// struct declaration
type Point struct {
    X int // public
    Y int // public
    internalVal int64 // private, start with lower case
    // no extra field can be added
}

// add method to struct outside struct
// golang use duck typing
func (destination Point) gridDist() int {
    return destination.X + destination.Y
}

func main() {
    var i1 int
    i1 = 3
    var i2 int32
    i2 = 99
    var i3 int64
    i3 = 99999999999999
    fmt.Println(i1)
    fmt.Println(i2)
    fmt.Println(i3)

    var f2 float32
    f2 = 0.3
    var f3 float64
    f3 = 123456789.123456789
    fmt.Println(f2)
    fmt.Println(f3)

    var s string
    s = "abc"
    fmt.Println(s)

    // arrays
    ns := []int{1,2,3}
    ss := []string {"aa", "b", "c"}
    fmt.Println(ns)
    fmt.Println(ss)

    // map
    var m map[string]int
    m = make(map[string]int) // map need initialization. nil map can read, not write
    m["aa"] = 2
    m["bb"] = 4
    print_map(m)

    // constructing and printing struct
    p := Point{X:3, Y:4, internalVal:9}
    fmt.Println(p)
    fmt.Println("grid dist is " + strconv.Itoa(p.gridDist()))
}
