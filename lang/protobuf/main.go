package main

import (
	"fmt"

	"github.com/golang/protobuf/jsonpb"

	geo "./geo"
)

func main() {
	c := geo.City{
		Name:       "Tokyo",
		Population: 10000000,
	}

	j, _ := (&jsonpb.Marshaler{}).MarshalToString(&c)

	fmt.Println("Name is", c.Name, "Population is", c.Population)
	fmt.Println("json is", j)
}
