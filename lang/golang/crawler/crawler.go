package main

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"

	"golang.org/x/net/html"
)

// Helper function to pull the href attribute from a Token
func getHref(t html.Token) (ok bool, href string) {
	// Iterate over all of the Token's attributes until we find an "href"
	for _, a := range t.Attr {
		if a.Key == "href" {
			href = a.Val
			ok = true
		}
	}

	// "bare" return will return the variables (ok, href) as defined in
	// the function definition
	return
}

func crawlOne(url string, chDone chan bool) {
	fmt.Println(url)

	resp, err := http.Get(url)

	// defer func() {
	// 	// Notify that we're done after this function
	// 	chDone <- true
	// }()

	if err != nil {
		fmt.Println("ERROR: Failed to crawl \"" + url + "\"")
		return
	}

	b := resp.Body
	defer b.Close() // close Body when the function returns

	z := html.NewTokenizer(b)

	done := false
	for {
		tt := z.Next()
		if done {
			break
		}

		switch {
		case tt == html.ErrorToken:
			// End of the document, we're done
			done = true
			break
		case tt == html.StartTagToken:
			t := z.Token()

			// Check if the token is an <a> tag
			isAnchor := t.Data == "a"
			if !isAnchor {
				continue
			}

			// Extract the href value, if there is one
			ok, url := getHref(t)
			if !ok {
				continue
			}

			// Make sure the url begines in http**
			hasProto := strings.Index(url, "http") == 0
			if hasProto {
				// chUrl <- url
				go crawlOne(url, chDone)
			}
		}
	}
	chDone <- true
}

func main() {
	seedUrl := os.Args[1]
	maxCrawl, _ := strconv.Atoi(string(os.Args[2]))

	// Channels
	// chUrls := make(chan string, 10000)
	chDone := make(chan bool)

	// go func() {
	// 	chUrls <- seedUrl
	// }()

	// Kick off the crawl process (concurrently)
	// for _, url := range seedUrls {
	//   go crawl(url, chUrls, chFinished)
	// }

	crawlOne(seedUrl, chDone)

	for i := 0; i < maxCrawl; i++ {
		// fmt.Println(i)
		<-chDone
	}

	// // Subscribe to both channels
	// for c := 0; c < len(seedUrls); {
	//   select {
	//   case url := <-chUrls:
	//     foundUrls[url] = true
	//   case <-chFinished:
	//     c++
	//   }
	// }
	//
	// // We're done! Print the results...
	//
	// fmt.Println("\nFound", len(foundUrls), "unique urls:\n")
	//
	// for url, _ := range foundUrls {
	//   fmt.Println(" - " + url)
	// }

	// close(chUrls)
}
