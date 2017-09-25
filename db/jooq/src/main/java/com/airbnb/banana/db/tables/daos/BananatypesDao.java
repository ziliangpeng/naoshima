/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db.tables.daos;


import com.airbnb.banana.db.tables.Bananatypes;
import com.airbnb.banana.db.tables.records.BananatypesRecord;

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
public class BananatypesDao extends DAOImpl<BananatypesRecord, com.airbnb.banana.db.tables.pojos.Bananatypes, Integer> {

    /**
     * Create a new BananatypesDao without any configuration
     */
    public BananatypesDao() {
        super(Bananatypes.BANANATYPES, com.airbnb.banana.db.tables.pojos.Bananatypes.class);
    }

    /**
     * Create a new BananatypesDao with an attached configuration
     */
    public BananatypesDao(Configuration configuration) {
        super(Bananatypes.BANANATYPES, com.airbnb.banana.db.tables.pojos.Bananatypes.class, configuration);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected Integer getId(com.airbnb.banana.db.tables.pojos.Bananatypes object) {
        return object.getId();
    }

    /**
     * Fetch records that have <code>id IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Bananatypes> fetchById(Integer... values) {
        return fetch(Bananatypes.BANANATYPES.ID, values);
    }

    /**
     * Fetch a unique record that has <code>id = value</code>
     */
    public com.airbnb.banana.db.tables.pojos.Bananatypes fetchOneById(Integer value) {
        return fetchOne(Bananatypes.BANANATYPES.ID, value);
    }

    /**
     * Fetch records that have <code>name IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Bananatypes> fetchByName(String... values) {
        return fetch(Bananatypes.BANANATYPES.NAME, values);
    }

    /**
     * Fetch records that have <code>price IN (values)</code>
     */
    public List<com.airbnb.banana.db.tables.pojos.Bananatypes> fetchByPrice(Integer... values) {
        return fetch(Bananatypes.BANANATYPES.PRICE, values);
    }
}
