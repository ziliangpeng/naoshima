package main

import (
  "context"
  "fmt"
  "log"
  "net"

  "./ig"
  "google.golang.org/grpc"
)

type InstagramServer struct {
}

func (s *InstagramServer) GetUser(ctx context.Context, getUserRequest *ig.GetUserRequest) (*ig.User, error) {
  u := ig.User{
    Id: "1",
    Username: "tokyo",
  }

  return &u, nil
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
