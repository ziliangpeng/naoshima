#!/bin/bash

run() {
  mysql -uroot -ppassword -e "$1" 2&>1 | grep -v "Using a password on the command line interface can be insecure"
}

run_seto() {
  mysql -uroot -ppassword seto -e "$1" 2&>1 | grep -v "Using a password on the command line interface can be insecure"
}

echo "Drop Database seto"
run "DROP DATABASE IF EXISTS seto;"

echo "Create Database seto"
run "CREATE DATABASE seto;" # Seto inland sea

echo "Create Table Islands"
run_seto "CREATE TABLE Islands (\
  id INT NOT NULL AUTO_INCREMENT, \
  name VARCHAR(64), \
  year INT, \
  population INT, \
  PRIMARY KEY (id));"

echo "Populate Table Islands"
run_seto "INSERT INTO Islands (id, name, year, population) VALUES (1, 'naoshima', 1987, 3000)"
run_seto "INSERT INTO Islands (id, name, year, population) VALUES (2, 'teshima', 1988, 2000)"
run_seto "INSERT INTO Islands (id, name, year, population) VALUES (3, 'shodoshima', 1979, 7000)"

echo "Create Table Arts"
run_seto "CREATE TABLE Arts (\
  id INT NOT NULL AUTO_INCREMENT, \
  island_id INT NOT NULL, \
  name VARCHAR(64), \
  price INT, \
  PRIMARY KEY (id), \
  FOREIGN KEY (island_id) REFERENCES Islands(id) \
  );"

echo "Populate Table Arts"
run_seto "INSERT INTO Arts (id, island_id, name, price) \
  VALUES (1, 1, 'Chi Chu Museum', 20)"
run_seto "INSERT INTO Arts (id, island_id, name, price) \
  VALUES (2, 2, 'Teshima Art Museum', 21)"
