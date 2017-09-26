package io.naoshima;

import io.naoshima.db.containers.DbQueryRequest;
import io.naoshima.db.listeners.AirbnbJooqListeners;
import org.jooq.Configuration;
import org.jooq.ConnectionProvider;
import org.jooq.DSLContext;
import org.jooq.SQLDialect;
import org.jooq.conf.RenderNameStyle;
import org.jooq.conf.Settings;
import org.jooq.conf.StatementType;
import org.jooq.impl.DSL;
import org.jooq.impl.DefaultConfiguration;
import org.jooq.impl.DefaultExecuteListenerProvider;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;

public class QueryExecutor {
    // TODO: include a list of default attributes e.g. retries, timeout, isTransaction,
    // The attributes will be used is absent from request object.

    private ConnectionProvider connectionProvider;

    private ExecutorService executorService;

    // TODO: private CircuitBreaker circuitBreaker;

    public QueryExecutor(ConnectionProvider connectionProvider, ExecutorService executorService) {
        this.connectionProvider = connectionProvider;
        this.executorService = executorService;

    }

    public QueryExecutor(ConnectionProvider connectionProvider) {
        this(connectionProvider, new ForkJoinPool());
    }

    // every `execute` should have an exclusive connection
    public <T> T execute(DbQueryRequest<T> dbQueryRequest) {
        DSLContext ctx = getDSLContext();
        return dbQueryRequest.getSql().apply(ctx);
    }

    // Shortcut for simply running a query with default configs
    public <T> T execute(Function<DSLContext, T> query) {
        DSLContext ctx = getDSLContext();
        return query.apply(ctx);
    }

    // TODO: implement async
    public <T> CompletableFuture<T> asyncExecute(DbQueryRequest<T> dbQueryRequest) {
        return CompletableFuture.supplyAsync(() -> execute(dbQueryRequest), executorService);
    }

    private DSLContext getDSLContext() {
        Configuration conf = new DefaultConfiguration()
                .derive(new Settings()
                        .withStatementType(StatementType.PREPARED_STATEMENT)
                        .withRenderNameStyle(RenderNameStyle.QUOTED))
                .set(connectionProvider.acquire())
                .set(SQLDialect.MYSQL)
                .set(new DefaultExecuteListenerProvider(new AirbnbJooqListeners()));
        return DSL.using(conf);
    }
}
