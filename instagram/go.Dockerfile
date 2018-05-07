FROM znly/protoc AS protoc

MAINTAINER Victor Peng version: 0.1

ADD ./proto /proto

ADD ./goscripts/gen_golang.sh /proto

RUN cd /proto && \
    mkdir ig && \
    sh ./gen_golang.sh



FROM golang:1.9-alpine AS golang

WORKDIR "/app"

RUN apk update && apk upgrade && \
    apk add --no-cache bash git openssh

ADD ./goscripts/*.go /app/

COPY --from=protoc /proto/ig/* /app/ig/

RUN cd /app && \
    go get -d ./... && \
    go build -o main .



FROM golang:1.9-alpine AS prod

WORKDIR "/app"

COPY --from=golang /app/main /app/main

#RUN ./main

ENTRYPOINT ["./main"]
