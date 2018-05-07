package main

import (
  "context"
  "encoding/json"
  "fmt"
  "log"
  "net"

  "./ig"
  "google.golang.org/grpc"
)

type InstagramServer struct {
}

func (s *InstagramServer) GetUser(ctx context.Context, getUserRequest *ig.GetUserRequest) (*ig.User, error) {
  fmt.Println("Request " + getUserRequest.Username)
  dat := readIgHTMLJSONDataBytes(getUserRequest.Username)
  var igProfil IGProfile
  json.Unmarshal([]byte(dat), &igProfil)

  igUser := igProfil.EntryData.ProfilePage[0].GraphQL.User
  var user ig.User
  user.Id = igUser.ID
  user.Username = igUser.Username
  user.FullName = igUser.Name
  fmt.Println(igUser)
  fmt.Println(user)
  return &user, nil
}

func startServer() {
  port := 8888
  lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
  if err != nil {
    log.Fatalf("failed to listen: %v", err)
  }
  grpcServer := grpc.NewServer()
  ig.RegisterInstagramServer(grpcServer, &InstagramServer{})
  fmt.Println("Serving...")
  grpcServer.Serve(lis)
}
