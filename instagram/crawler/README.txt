# To crawl Instagram

idea documents in Day One note.



## MySQL schema

CREATE TABLE `user` (
  `id` bigint(11) unsigned NOT NULL AUTO_INCREMENT,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  `user_id` bigint(11) unsigned NOT NULL,
  `username` varchar(255) NOT NULL,
  `biography` text,
  `external_url` text,
  `followed_by_count` int,
  `follows_count` int,
  `full_name` text,
  `is_private` bool,
  `is_verified` bool,
  `profile_pic_url_hd` varchar(256),
  `claimed` DATETIME,
  `done` DATETIME
  PRIMARY KEY (`id`),
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `images` (
  `id` bigint(11) unsigned NOT NULL AUTO_INCREMENT,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  `type` varchar(64),
  `image_id` varchar(64),
  `code` varchar(64),
  `height` smallint unsigned,
  `width` smallint unsigned,
  `display_url` text,
  `is_video` bool,
  `caption` text,
  `temp_likes` int
  `claimed` DATETIME  # for worker to claim a task
  `done` DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

# mat not need this with `done` column
CREATE TABLE `likes` (
  `id` bigint(11) unsigned NOT NULL AUTO_INCREMENT,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  `image_id` bigint
  `user_id` bigint
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



CREATE TABLE `history` (
  `id` bigint(11) unsigned NOT NULL AUTO_INCREMENT,
  `time` datetime,
  `user_count` int,
  `crawled_user_count` int,
  `private_user_count` int,
  `image_count` int,
  `mysql_connections` int
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
