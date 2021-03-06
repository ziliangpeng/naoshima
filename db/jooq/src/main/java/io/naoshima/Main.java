package io.naoshima;

import com.airbnb.banana.db.Tables;
import com.airbnb.banana.db.tables.records.UsersRecord;
import io.naoshima.db.containers.DbQueryRequest;
import io.naoshima.db.containers.DbTransactionRequest;
import org.jooq.ConnectionProvider;
import org.jooq.DSLContext;
import org.jooq.impl.DSL;
import org.jooq.impl.DefaultConnectionProvider;

import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class Main {

  public static void main(String argv[]) throws InterruptedException, SQLException {

    QueryExecutor executor = new QueryExecutor(getConnectionProvider(), new ForkJoinPool(500));

    DbQueryRequest<UsersRecord> request = new DbQueryRequest<>(ctx -> ctx.selectFrom(Tables.USERS).where(true).fetchAny());

    // Query 1
    UsersRecord r1 = executor.execute(request);
    System.out.println("Name of user is: " + r1.getName());

    // Query 2
    UsersRecord r2 = executor.execute(ctx ->
            ctx
            .selectFrom(Tables.USERS)
            .where(Tables.USERS.WEALTH.greaterThan(5000))
            .fetchAny());
    System.out.println("Name of user is: " + r2.getName());

    // Multiple async queries
    IntStream.range(1, 100).mapToObj(i -> executor.asyncExecute(request)).forEach(cf -> cf.join());

    // Transaction
    DbTransactionRequest transactionRequest = new DbTransactionRequest(conf -> {
      DSLContext ctx = DSL.using(conf);
      UsersRecord a = ctx.selectFrom(Tables.USERS).where(Tables.USERS.NAME.eq("A")).fetchOne();
      UsersRecord b = ctx.selectFrom(Tables.USERS).where(Tables.USERS.NAME.eq("B")).fetchOne();
      System.out.println("Wealth is " + a.getWealth() + " " + b.getWealth());
      a.setWealth(a.getWealth() - 100);
      b.setWealth(b.getWealth() + 100);
      a.store();
      b.store();
    });
    executor.asyncExecuteTransaction(transactionRequest);

    // Multiple async transaction (will fail because race transactions)
    IntStream.range(1, 100).mapToObj(i -> executor.asyncExecuteTransaction(transactionRequest)).forEach(cf -> cf.join());
  }

  private static ConnectionProvider getConnectionProvider() throws SQLException {
    // TODO: current impl is a naive single connection provider. To use DataSourceConnectionProvider and
    // Dropwizard's DatasourceFactory

    String userName = "root";
    String password = "password";
    String url = "jdbc:mysql://localhost:3306/banana";
    return new DefaultConnectionProvider(DriverManager.getConnection(url, userName, password));
  }
}
