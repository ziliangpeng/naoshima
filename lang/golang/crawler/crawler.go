package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"

	"./util"
)

func mergeQueue(queue chan string, links []string) {
	for _, url := range links {
		select {
		case queue <- url:
		default:
		}
	}
}

func crawlOne(queue chan string, url string) {
	fmt.Println("Crawling", url)

	resp, err := http.Get(url)

	if err != nil {
		fmt.Println("ERROR: Failed to crawl \"" + url + "\"")
		return
	}

	if resp.StatusCode != 200 {
		fmt.Println("Not 200:", resp.StatusCode, url)
	}

	b := resp.Body
	defer b.Close() // close Body when the function returns

	links := util.ParseLinks(b)
	io.Copy(ioutil.Discard, resp.Body)
	mergeQueue(queue, links)

	return
}

func main() {
	seedUrl := os.Args[1]
	maxCrawl, _ := strconv.Atoi(string(os.Args[2]))
	queue := make(chan string, maxCrawl*10)
	mergeQueue(queue, []string{seedUrl})

	for i := 0; i < maxCrawl; i++ {
		go func() {
			for {
				url := <-queue
				crawlOne(queue, url)
			}
		}()
	}

	ch := make(chan int)
	<-ch
}
