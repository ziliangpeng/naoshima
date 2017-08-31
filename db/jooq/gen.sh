#!/bin/bash

java -cp jooq-3.9.5.jar:jooq-codegen-3.9.5.jar:jooq-meta-3.9.5.jar:./mysql-connector-java-5.1.44/mysql-connector-java-5.1.44-bin.jar org.jooq.util.GenerationTool jooq_config.xml
