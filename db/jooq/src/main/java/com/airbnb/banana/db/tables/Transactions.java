/*
 * This file is generated by jOOQ.
*/
package com.airbnb.banana.db.tables;


import com.airbnb.banana.db.DefaultSchema;
import com.airbnb.banana.db.Keys;
import com.airbnb.banana.db.tables.records.TransactionsRecord;

import java.util.Arrays;
import java.util.List;

import javax.annotation.Generated;

import org.jooq.Field;
import org.jooq.ForeignKey;
import org.jooq.Identity;
import org.jooq.Schema;
import org.jooq.Table;
import org.jooq.TableField;
import org.jooq.UniqueKey;
import org.jooq.impl.TableImpl;


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
public class Transactions extends TableImpl<TransactionsRecord> {

    private static final long serialVersionUID = -2110411190;

    /**
     * The reference instance of <code>Transactions</code>
     */
    public static final Transactions TRANSACTIONS = new Transactions();

    /**
     * The class holding records for this type
     */
    @Override
    public Class<TransactionsRecord> getRecordType() {
        return TransactionsRecord.class;
    }

    /**
     * The column <code>Transactions.id</code>.
     */
    public final TableField<TransactionsRecord, Integer> ID = createField("id", org.jooq.impl.SQLDataType.INTEGER.nullable(false), this, "");

    /**
     * The column <code>Transactions.sender</code>.
     */
    public final TableField<TransactionsRecord, Integer> SENDER = createField("sender", org.jooq.impl.SQLDataType.INTEGER, this, "");

    /**
     * The column <code>Transactions.receiver</code>.
     */
    public final TableField<TransactionsRecord, Integer> RECEIVER = createField("receiver", org.jooq.impl.SQLDataType.INTEGER, this, "");

    /**
     * The column <code>Transactions.amount</code>.
     */
    public final TableField<TransactionsRecord, Integer> AMOUNT = createField("amount", org.jooq.impl.SQLDataType.INTEGER, this, "");

    /**
     * Create a <code>Transactions</code> table reference
     */
    public Transactions() {
        this("Transactions", null);
    }

    /**
     * Create an aliased <code>Transactions</code> table reference
     */
    public Transactions(String alias) {
        this(alias, TRANSACTIONS);
    }

    private Transactions(String alias, Table<TransactionsRecord> aliased) {
        this(alias, aliased, null);
    }

    private Transactions(String alias, Table<TransactionsRecord> aliased, Field<?>[] parameters) {
        super(alias, null, aliased, parameters, "");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Schema getSchema() {
        return DefaultSchema.DEFAULT_SCHEMA;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Identity<TransactionsRecord, Integer> getIdentity() {
        return Keys.IDENTITY_TRANSACTIONS;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public UniqueKey<TransactionsRecord> getPrimaryKey() {
        return Keys.KEY_TRANSACTIONS_PRIMARY;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<UniqueKey<TransactionsRecord>> getKeys() {
        return Arrays.<UniqueKey<TransactionsRecord>>asList(Keys.KEY_TRANSACTIONS_PRIMARY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<ForeignKey<TransactionsRecord, ?>> getReferences() {
        return Arrays.<ForeignKey<TransactionsRecord, ?>>asList(Keys.TRANSACTIONS_IBFK_1, Keys.TRANSACTIONS_IBFK_2);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Transactions as(String alias) {
        return new Transactions(alias, this);
    }

    /**
     * Rename this table
     */
    @Override
    public Transactions rename(String name) {
        return new Transactions(name, null);
    }
}