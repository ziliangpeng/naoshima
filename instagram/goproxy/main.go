package main

import (
	"net/http"
	"strings"
)

func sayHello(w http.ResponseWriter, r *http.Request) {
	newline := "\n"
	name := strings.TrimPrefix(r.URL.Path, "/")
	message := "Your name is " + name + newline
	message += "Your agent is " + r.UserAgent()

	w.Write([]byte(message))
}

func main() {
	http.HandleFunc("/", sayHello)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		panic(err)
	}
}
