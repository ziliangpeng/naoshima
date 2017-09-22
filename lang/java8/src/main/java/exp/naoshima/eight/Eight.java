package exp.naoshima.eight;

import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

class Eight {

    // shorter method to reference System.out::println
    private final static Consumer<String> p = System.out::println;
    public static void main(String[] args) {
        int value = 0;

        // Functionally reduce. Apply lambda to an initial value n times
        value = IntStream.range(0, 4).reduce(value, (a, b) -> manipulateInt(a, v -> v + 1));

        value = IntStream.range(0, 4).reduce(value, (a, b) -> manipulateInt(a, v -> v * 2));

        p.accept("Final value is " + value);

        doRun(() -> {p.accept("Inside runner");});
    }

    /*
     * Apply a lambda (op) to a value
     */
    private static int manipulateInt(int value, IntUnaryOperator op) {
        p.accept("Before value: " + value);
        int ret = op.applyAsInt(value);
        p.accept("After value:  " + ret);
        p.accept("=====");
        return ret;
    }

    private static void doRun(NaoshimaRunner r) {
      p.accept("Starting to run runner");
      r.runMethod();
      p.accept("Finished run runner");
    }
}
