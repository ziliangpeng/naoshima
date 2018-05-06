package main

import (
  "encoding/json"
  "fmt"
  "io/ioutil"
)

// User: Instagram User struct
type User struct {
  Biography string `json:"biography"`
}

func test() {
  filename := "ig_user_data.json"
  bytes, _ := ioutil.ReadFile(filename)
  var dat map[string]interface{}
  json.Unmarshal(bytes, &dat)
  datUser := getMapListKeyGeneral(dat, "graphql", "user")
  var u User
  userBytes, _ := json.Marshal(datUser)
  json.Unmarshal([]byte(userBytes), &u)
  fmt.Println(u)
  fmt.Println(u.Biography)
}
