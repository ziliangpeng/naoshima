package io.naoshima.db.containers;

import org.jooq.DSLContext;

import java.util.function.Function;

public class DbQueryRequest<T> {
    // TODO: a list of attributes e.g. timeout, retries, isTransaction,
    public Function<DSLContext, T> getSql() {
        return sql;
    }

    private Function<DSLContext, T> sql;

    public DbQueryRequest(Function<DSLContext, T> sql) {
        this.sql = sql;
    }
}
