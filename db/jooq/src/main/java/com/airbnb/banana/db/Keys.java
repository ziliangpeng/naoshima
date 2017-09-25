/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db;


import com.airbnb.banana.db.tables.Bananatypes;
import com.airbnb.banana.db.tables.Transactions;
import com.airbnb.banana.db.tables.Users;
import com.airbnb.banana.db.tables.records.BananatypesRecord;
import com.airbnb.banana.db.tables.records.TransactionsRecord;
import com.airbnb.banana.db.tables.records.UsersRecord;

import javax.annotation.Generated;

import org.jooq.ForeignKey;
import org.jooq.Identity;
import org.jooq.UniqueKey;
import org.jooq.impl.AbstractKeys;


/**
 * A class modelling foreign key relationships between tables of the <code></code> 
 * schema
 */
@Generated(
    value = {
        "http://www.jooq.org",
        "jOOQ version:3.9.5"
    },
    comments = "This class is generated by jOOQ"
)
@SuppressWarnings({ "all", "unchecked", "rawtypes" })
public class Keys {

    // -------------------------------------------------------------------------
    // IDENTITY definitions
    // -------------------------------------------------------------------------

    public static final Identity<BananatypesRecord, Integer> IDENTITY_BANANATYPES = Identities0.IDENTITY_BANANATYPES;
    public static final Identity<TransactionsRecord, Integer> IDENTITY_TRANSACTIONS = Identities0.IDENTITY_TRANSACTIONS;
    public static final Identity<UsersRecord, Integer> IDENTITY_USERS = Identities0.IDENTITY_USERS;

    // -------------------------------------------------------------------------
    // UNIQUE and PRIMARY KEY definitions
    // -------------------------------------------------------------------------

    public static final UniqueKey<BananatypesRecord> KEY_BANANATYPES_PRIMARY = UniqueKeys0.KEY_BANANATYPES_PRIMARY;
    public static final UniqueKey<TransactionsRecord> KEY_TRANSACTIONS_PRIMARY = UniqueKeys0.KEY_TRANSACTIONS_PRIMARY;
    public static final UniqueKey<UsersRecord> KEY_USERS_PRIMARY = UniqueKeys0.KEY_USERS_PRIMARY;

    // -------------------------------------------------------------------------
    // FOREIGN KEY definitions
    // -------------------------------------------------------------------------

    public static final ForeignKey<TransactionsRecord, UsersRecord> TRANSACTIONS_IBFK_1 = ForeignKeys0.TRANSACTIONS_IBFK_1;
    public static final ForeignKey<TransactionsRecord, UsersRecord> TRANSACTIONS_IBFK_2 = ForeignKeys0.TRANSACTIONS_IBFK_2;

    // -------------------------------------------------------------------------
    // [#1459] distribute members to avoid static initialisers > 64kb
    // -------------------------------------------------------------------------

    private static class Identities0 extends AbstractKeys {
        public static Identity<BananatypesRecord, Integer> IDENTITY_BANANATYPES = createIdentity(Bananatypes.BANANATYPES, Bananatypes.BANANATYPES.ID);
        public static Identity<TransactionsRecord, Integer> IDENTITY_TRANSACTIONS = createIdentity(Transactions.TRANSACTIONS, Transactions.TRANSACTIONS.ID);
        public static Identity<UsersRecord, Integer> IDENTITY_USERS = createIdentity(Users.USERS, Users.USERS.ID);
    }

    private static class UniqueKeys0 extends AbstractKeys {
        public static final UniqueKey<BananatypesRecord> KEY_BANANATYPES_PRIMARY = createUniqueKey(Bananatypes.BANANATYPES, "KEY_BananaTypes_PRIMARY", Bananatypes.BANANATYPES.ID);
        public static final UniqueKey<TransactionsRecord> KEY_TRANSACTIONS_PRIMARY = createUniqueKey(Transactions.TRANSACTIONS, "KEY_Transactions_PRIMARY", Transactions.TRANSACTIONS.ID);
        public static final UniqueKey<UsersRecord> KEY_USERS_PRIMARY = createUniqueKey(Users.USERS, "KEY_Users_PRIMARY", Users.USERS.ID);
    }

    private static class ForeignKeys0 extends AbstractKeys {
        public static final ForeignKey<TransactionsRecord, UsersRecord> TRANSACTIONS_IBFK_1 = createForeignKey(com.airbnb.banana.db.Keys.KEY_USERS_PRIMARY, Transactions.TRANSACTIONS, "transactions_ibfk_1", Transactions.TRANSACTIONS.SENDER);
        public static final ForeignKey<TransactionsRecord, UsersRecord> TRANSACTIONS_IBFK_2 = createForeignKey(com.airbnb.banana.db.Keys.KEY_USERS_PRIMARY, Transactions.TRANSACTIONS, "transactions_ibfk_2", Transactions.TRANSACTIONS.RECEIVER);
    }
}
