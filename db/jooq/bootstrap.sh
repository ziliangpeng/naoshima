#!/bin/bash

run() {
  mysql -uroot -ppassword -e "$1" 2>&1 | grep -v "Using a password on the command line interface can be insecure"
}

run_banana() {
  mysql -uroot -ppassword banana -e "$1" 2>&1 | grep -v "Using a password on the command line interface can be insecure"
}

echo "Drop Database banana"
run "DROP DATABASE IF EXISTS banana;"

echo "Create Database banana"
run "CREATE DATABASE banana;"


echo "Create Table Users"
run_banana "CREATE TABLE Users (\
  id INT NOT NULL AUTO_INCREMENT, \
  name VARCHAR(64), \
  wealth INT, \
  PRIMARY KEY (id));"

echo "Populate Table Users"
run_banana "INSERT INTO Users (id, name, wealth) VALUES (1, 'A', 3000)"
run_banana "INSERT INTO Users (id, name, wealth) VALUES (2, 'B', 2000)"
run_banana "INSERT INTO Users (id, name, wealth) VALUES (3, 'C', 7000)"


echo "Create Table BananaTypes"
run_banana "CREATE TABLE BananaTypes (\
  id INT NOT NULL AUTO_INCREMENT, \
  name VARCHAR(64), \
  price INT, \
  PRIMARY KEY (id) \
  );"

echo "Populate Table BananaTypes"
run_banana "INSERT INTO BananaTypes (id, name, price) \
  VALUES (1, 'Simple Banana', 20);"


echo "Create Table Transactions"
run_banana "CREATE TABLE Transactions (\
  id INT NOT NULL AUTO_INCREMENT, \
  sender INT, \
  receiver INT, \
  amount INT, \
  FOREIGN KEY (sender) REFERENCES Users(id), \
  FOREIGN KEY (receiver) REFERENCES Users(id), \
  PRIMARY KEY (id) \
  );"
