package main

import (
    "fmt"
    "net/http"
    "io/ioutil"
)

func handler(w http.ResponseWriter, r *http.Request) {
    username := r.URL.Path[1:]
    url := fmt.Sprintf("https://www.instagram.com/%s/?__a=1", username)
    resp, _ := http.Get(url)

    defer resp.Body.Close()
    body, _ := ioutil.ReadAll(resp.Body)

    fmt.Printf(string(body))
    fmt.Fprintf(w, string(body))
}

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8080", nil)
}
