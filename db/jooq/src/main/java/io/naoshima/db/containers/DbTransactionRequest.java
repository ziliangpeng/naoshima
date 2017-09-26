package io.naoshima.db.containers;

import org.jooq.Configuration;

import java.util.function.Consumer;

public class DbTransactionRequest {
    // TODO: a list of attributes e.g. timeout, retries, isTransaction,
    public Consumer<Configuration> getTransaction() {
        return transaction;
    }

    private Consumer<Configuration> transaction;

    public DbTransactionRequest(Consumer<Configuration> transaction) {
        this.transaction = transaction;
    }
}
