package exp.naoshima.eight;

import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

class Eight {

    public static void main(String[] args) {
        int value = 0;

        // Functionally reduce. Apply lambda to an initial value n times
        value = IntStream.range(0, 4).reduce(value, (a, b) -> manipulateInt(a, v -> v + 1));

        value = IntStream.range(0, 4).reduce(value, (a, b) -> manipulateInt(a, v -> v * 2));

        System.out.println("Final value is " + value);
    }

    /*
     * Apply a lambda (op) to a value
     */
    private static int manipulateInt(int value, IntUnaryOperator op) {
        System.out.println("Before value: " + value);
        int ret = op.applyAsInt(value);
        System.out.println("After value:  " + ret);
        System.out.println("=====");
        return ret;
    }
}
