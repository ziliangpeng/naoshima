/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db.tables.pojos;


import java.io.Serializable;

import javax.annotation.Generated;


/**
 * This class is generated by jOOQ.
 */
@Generated(
    value = {
        "http://www.jooq.org",
        "jOOQ version:3.9.5"
    },
    comments = "This class is generated by jOOQ"
)
@SuppressWarnings({ "all", "unchecked", "rawtypes" })
public class Users implements Serializable {

    private static final long serialVersionUID = -939081561;

    private Integer id;
    private String  name;
    private Integer wealth;

    public Users() {}

    public Users(Users value) {
        this.id = value.id;
        this.name = value.name;
        this.wealth = value.wealth;
    }

    public Users(
        Integer id,
        String  name,
        Integer wealth
    ) {
        this.id = id;
        this.name = name;
        this.wealth = wealth;
    }

    public Integer getId() {
        return this.id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getWealth() {
        return this.wealth;
    }

    public void setWealth(Integer wealth) {
        this.wealth = wealth;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Users (");

        sb.append(id);
        sb.append(", ").append(name);
        sb.append(", ").append(wealth);

        sb.append(")");
        return sb.toString();
    }
}
