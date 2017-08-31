#!/bin/bash

echo "Drop Database seto"
mysql -uroot -e "DROP DATABASE IF EXISTS seto;"

echo "Create Database seto"
mysql -uroot -e "CREATE DATABASE seto;" # Seto inland sea

echo "Create Table Islands"
mysql -uroot seto -e "CREATE TABLE Islands (\
  id INT NOT NULL AUTO_INCREMENT, \
  name VARCHAR(64), \
  year INT, \
  population INT, \
  PRIMARY KEY (id));"

echo "Populate Table Islands"
mysql -uroot seto -e "INSERT INTO Islands (id, name, year, population) VALUES (1, 'naoshima', 1987, 3000)"
mysql -uroot seto -e "INSERT INTO Islands (id, name, year, population) VALUES (2, 'teshima', 1988, 2000)"
mysql -uroot seto -e "INSERT INTO Islands (id, name, year, population) VALUES (3, 'shodoshima', 1979, 7000)"

echo "Create Table Arts"
mysql -uroot seto -e "CREATE TABLE Arts (\
  id INT NOT NULL AUTO_INCREMENT, \
  island_id INT NOT NULL, \
  name VARCHAR(64), \
  price INT, \
  PRIMARY KEY (id), \
  FOREIGN KEY (island_id) REFERENCES Islands(id) \
  );"

echo "Populate Table Arts"
mysql -uroot seto -e "INSERT INTO Arts (id, island_id, name, price) \
  VALUES (1, 1, 'Chi Chu Museum', 20)"
mysql -uroot seto -e "INSERT INTO Arts (id, island_id, name, price) \
  VALUES (2, 2, 'Teshima Art Museum', 21)"
