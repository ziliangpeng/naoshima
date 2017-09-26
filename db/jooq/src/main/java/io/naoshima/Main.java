package io.naoshima;

import com.airbnb.banana.db.Tables;
import com.airbnb.banana.db.tables.records.UsersRecord;
import io.naoshima.db.containers.DbQueryRequest;
import org.jooq.ConnectionProvider;
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

    // Multiple queries
    IntStream.range(1, 100).mapToObj(i -> executor.asyncExecute(request)).forEach(cf -> cf.join());
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
