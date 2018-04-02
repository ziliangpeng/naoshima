package main

import (
	"github.com/go-redis/redis"
)

const REDIS_DB = 1

const NAMESPACE_USER_LIKED = "total_likes:"

func createRedis(redisHost string, redisPort string, redisDB int) *redis.Client {
	return redis.NewClient(&redis.Options{
		Addr:     redisHost + ":" + redisPort,
		Password: "", // no password set
		DB:       redisDB,
	})
}

func saveUsereLike(rc *redis.Client, username string, time int64, likes int) {
	rc.HSet(NAMESPACE_USER_LIKED+username, string(time), likes)
}
