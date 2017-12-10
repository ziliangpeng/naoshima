# To crawl Instagram

idea documents in Day One note.



## MySQL schema

CREATE TABLE `instagram_user` (
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
  PRIMARY KEY (`id`),
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `instagram_user` (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
