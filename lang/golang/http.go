package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path[1:]
	query := r.URL.Query()
	url := fmt.Sprintf("https://www.instagram.com/%s?%s", path, query)
	resp, _ := http.Get(url)

	defer resp.Body.Close()
	body, _ := ioutil.ReadAll(resp.Body)

	fmt.Println(url)
	fmt.Fprintf(w, string(body))
}

func main() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
