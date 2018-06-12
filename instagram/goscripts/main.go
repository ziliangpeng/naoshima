package main

import (
  "encoding/json"
  "flag"
  "fmt"
  "io/ioutil"
  "net/http"
  "reflect"
  "strings"
)

// works only for dict
func getMapKeyGeneralDeprecated(m interface{}, keys ...string) interface{} {
  for _, key := range keys {
    m = m.(map[string]interface{})[key]
  }
  return m
}

// works better for json, support dict key reading and list index reading
func getMapListKeyGeneral(m interface{}, keys ...interface{}) interface{} {
  for _, key := range keys {
    if reflect.TypeOf(key).String() == "string" {
      m = m.(map[string]interface{})[key.(string)]
    } else if reflect.TypeOf(key).String() == "int" {
      m = m.([]interface{})[key.(int)]
    }
  }
  return m
}

func readIgHTMLJSONDataBytes(username string) string {
  resp, err := http.Get(fmt.Sprintf("https://www.instagram.com/%s/", username))
  defer resp.Body.Close()
  if err != nil {
  } // handle error
  bodyBytes, err := ioutil.ReadAll(resp.Body)
  if err != nil {
  } // handle error

  bodyString := string(bodyBytes)
  // fmt.Println(bodyString)
  lines := strings.Split(bodyString, "\n")
  line := lines[236]
  // fmt.Println(line)
  startIndex := strings.Index(line, "{")
  endIndex := strings.LastIndex(line, "}") + 1
  return line[startIndex:endIndex]
}

// IG changed protocol. read from HTML
func readJsonFromIgHTML(username string) map[string]interface{} {
  j:= readIgHTMLJSONDataBytes(username)

  var dat map[string]interface{}
  json.Unmarshal([]byte(j), &dat)
  return dat
}

// old IG protocol. Not work any more.
func readJsonFromIgDeprecated(username string) map[string]interface{} {
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
  dat := readJsonFromIgHTML(username)
  datJSON := getMapListKeyGeneral(dat, "entry_data", "ProfilePage", 0)
  followedBy := getMapListKeyGeneral(datJSON, "graphql", "user", "edge_followed_by", "count").(float64)
  follow := getMapListKeyGeneral(datJSON, "graphql", "user", "edge_follow", "count").(float64)

  fmt.Println(int64(followedBy), int64(follow))
}

func readLikesStats(username string) {
  dat := readJsonFromIgHTML(username)
  datJSON := getMapListKeyGeneral(dat, "entry_data", "ProfilePage", 0)
  media := getMapListKeyGeneral(datJSON, "graphql", "user", "edge_owner_to_timeline_media", "edges").([]interface{})
  likes := int64(0)
  for _, entry := range media {
    likes += int64(getMapListKeyGeneral(entry.(map[string]interface{}), "node", "edge_liked_by", "count").(float64))
  }
  fmt.Println(likes)
}

func main() {
  var task string
  var username string
  var saveToRedis bool
  flag.StringVar(&task, "t", "likes", "task to scripting")
  flag.StringVar(&username, "u", "instagram", "username to scripting")
  flag.BoolVar(&saveToRedis, "r", false, "save result to Redis")
  flag.Parse()

  if task == "likes" {
    readLikesStats(username)
  } else if task == "follow" {
    readFollowStats(username)
  } else if task == "model" {
    test()
  } else if task == "proxy" {
    startServer()
  } else {
    panic("wrong task type " + task)
  }
}
