package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
)

func getMapKeyGeneral(m interface{}, keys ...string) interface{} {
	for _, key := range keys {
		m = m.(map[string]interface{})[key]
	}
	return m
}

func readJsonFromIg(username string) map[string]interface{} {
	resp, err := http.Get(fmt.Sprintf("https://www.instagram.com/%s/?__a=1", username))
	defer resp.Body.Close()
	if err != nil {
	} // handle error
	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
	} // handle error

	var dat map[string]interface{}
	json.Unmarshal(bodyBytes, &dat)
	return dat
}

func readFollowStats(username string) {
	dat := readJsonFromIg(username)
	followedBy := getMapKeyGeneral(dat, "graphql", "user", "edge_followed_by", "count").(float64)
	follow := getMapKeyGeneral(dat, "graphql", "user", "edge_follow", "count").(float64)

	fmt.Println(int64(followedBy))
	fmt.Println(int64(follow))
}

func readLikesStats(username string) {
	dat := readJsonFromIg(username)
	media := getMapKeyGeneral(dat, "graphql", "user", "edge_owner_to_timeline_media", "edges").([]interface{})
	likes := int64(0)
	for _, entry := range media {
		likes += int64(getMapKeyGeneral(entry.(map[string]interface{}), "node", "edge_liked_by", "count").(float64))
	}
	fmt.Println(likes)
}

func main() {
	var task string
	var username string
	flag.StringVar(&task, "t", "likes", "task to scripting")
	flag.StringVar(&username, "u", "instagram", "username to scripting")
	flag.Parse()

	if task == "likes" {
		readLikesStats(username)
	} else if task == "follow" {
		readFollowStats(username)
	} else {
		panic("wrong task type " + task)
	}
}
