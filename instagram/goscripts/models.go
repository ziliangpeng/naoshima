package main

import (
  "encoding/json"
  "fmt"
  "io/ioutil"
)

type IGMedia struct {
  ID string `json:"id"`
  Code string `json:"shortcode"`
  Type string `json:"__typename"`
  Owner struct {
    ID string `json:"id"`
  } `json:"owner"`
  Dimentions struct {
    Height int `json:"height"`
    Width int `json:"width"`
  } `json:"dimensions"`
  Timestamp int `json:"taken_at_timestamp"`
  LikedBy struct {
    Count int `json:"count"`
  } `json:"edge_liked_by"`
}

type IGUser struct {
  ID string `json:"id"`
  Name string `json:"full_name"`
  Username string `json:"username"`
  Biography string `json:"biography"`
  IsPrivate bool `json:"is_private"`
  IsVerified bool `json:"is_verified"`

  FollowedBy struct {
    Count int `json:"count"`
  } `json:"edge_followed_by"`

  Follows struct {
    Count int `json:"count"`
  } `json:"edge_follow"`

  Medias struct {
    Count int `json:"count"`
    MediaEdges []struct {
      Media IGMedia `json:"node"`
    } `json:"edges"`
  } `json:"edge_owner_to_timeline_media"`
}

type IGProfile struct {
  EntryData struct {
    ProfilePage []struct {
      GraphQL struct {
        User IGUser `json:"user"`
      } `json:"graphql"`
    } `json:"ProfilePage"`
  } `json:"entry_data"`
}


func test() {
  filename := "ig_user_profile.json"
  bytes, _ := ioutil.ReadFile(filename)

  var igProfile IGProfile

  json.Unmarshal([]byte(bytes), &igProfile)

  user := igProfile.EntryData.ProfilePage[0].GraphQL.User
  fmt.Printf("%+v\n\n", user)
  j, _ := json.Marshal(user)
  fmt.Println(string(j))
}
