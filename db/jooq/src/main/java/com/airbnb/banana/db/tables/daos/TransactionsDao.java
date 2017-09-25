/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db.tables.daos;


import com.airbnb.banana.db.tables.Transactions;
import com.airbnb.banana.db.tables.records.TransactionsRecord;

import java.util.List;

import javax.annotation.Generated;

import org.jooq.Configuration;
import org.jooq.impl.DAOImpl;


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
public class TransactionsDao extends DAOImpl<TransactionsRecord, com.airbnb.banana.db.tables.pojos.Transactions, Integer> {

    /**
     * Create a new TransactionsDao without any configuration
     */
    public TransactionsDao() {
        super(Transactions.TRANSACTIONS, com.airbnb.banana.db.tables.pojos.Transactions.class);
    }

    /**
     * Create a new TransactionsDao with an attached configuration
     */
    public TransactionsDao(Configuration configuration) {
        super(Transactions.TRANSACTIONS, com.airbnb.banana.db.tables.pojos.Transactions.class, configuration);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected Integer getId(com.airbnb.banana.db.tables.pojos.Transactions object) {
        return object.getId();
    }

    /**
     * Fetch records that have <code>id IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Transactions> fetchById(Integer... values) {
        return fetch(Transactions.TRANSACTIONS.ID, values);
    }

    /**
     * Fetch a unique record that has <code>id = value</code>
     */
    public com.airbnb.banana.db.tables.pojos.Transactions fetchOneById(Integer value) {
        return fetchOne(Transactions.TRANSACTIONS.ID, value);
    }

    /**
     * Fetch records that have <code>sender IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Transactions> fetchBySender(Integer... values) {
        return fetch(Transactions.TRANSACTIONS.SENDER, values);
    }

    /**
     * Fetch records that have <code>receiver IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Transactions> fetchByReceiver(Integer... values) {
        return fetch(Transactions.TRANSACTIONS.RECEIVER, values);
    }

    /**
     * Fetch records that have <code>amount IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Transactions> fetchByAmount(Integer... values) {
        return fetch(Transactions.TRANSACTIONS.AMOUNT, values);
    }
}
