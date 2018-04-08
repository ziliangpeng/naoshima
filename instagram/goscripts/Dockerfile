FROM golang:1.9-stretch

MAINTAINER Victor Peng version: 0.1

ADD . /app

WORKDIR "/app"

RUN go build main.go

ENTRYPOINT ["./main"]
