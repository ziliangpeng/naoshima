/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db;


import com.airbnb.banana.db.tables.Bananatypes;
import com.airbnb.banana.db.tables.Transactions;
import com.airbnb.banana.db.tables.Users;

import javax.annotation.Generated;


/**
 * Convenience access to all tables in 
 */
@Generated(
    value = {
        "http://www.jooq.org",
        "jOOQ version:3.9.5"
    },
    comments = "This class is generated by jOOQ"
)
@SuppressWarnings({ "all", "unchecked", "rawtypes" })
public class Tables {

    /**
     * The table <code>BananaTypes</code>.
     */
    public static final Bananatypes BANANATYPES = com.airbnb.banana.db.tables.Bananatypes.BANANATYPES;

    /**
     * The table <code>Transactions</code>.
     */
    public static final Transactions TRANSACTIONS = com.airbnb.banana.db.tables.Transactions.TRANSACTIONS;

    /**
     * The table <code>Users</code>.
     */
    public static final Users USERS = com.airbnb.banana.db.tables.Users.USERS;
}