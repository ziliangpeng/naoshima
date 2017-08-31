#!/bin/bash


echo "drop database seto"
mysql -uroot -e "DROP DATABASE IF EXISTS seto;"

echo "create database seto"
mysql -uroot -e "CREATE DATABASE seto;" # Seto inland sea

echo "create table islands"
mysql -uroot seto -e "CREATE TABLE islands (name VARCHAR(64), year INT, population INT);"

echo "populate data to islands"
mysql -uroot seto -e "INSERT INTO islands (name, year, population) VALUES ('naoshima', 1987, 3000)"
mysql -uroot seto -e "INSERT INTO islands (name, year, population) VALUES ('teshima', 1988, 2000)"
mysql -uroot seto -e "INSERT INTO islands (name, year, population) VALUES ('shodoshima', 1979, 7000)"
