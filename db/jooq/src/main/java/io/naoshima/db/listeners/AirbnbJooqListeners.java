package io.naoshima.db.listeners;

import org.jooq.ExecuteContext;
import org.jooq.impl.DefaultExecuteListener;

public class AirbnbJooqListeners extends DefaultExecuteListener {

    @Override
    public void executeStart(ExecuteContext ctx) {
        System.out.println("listening executeStart");
        String sql = ctx.sql();
        System.out.println("original sql is: " + sql);

        sql = "/* SOME COMMENTS */ " + sql;
        System.out.println("modified sql is: " + sql);

        ctx.sql(sql);
    }

    public void executeEnd(ExecuteContext ctx) {
        System.out.println("listening executeEnd");
        String sql = ctx.sql();
        System.out.println("executed sql is: " + sql);
    }
}
