template <class T>
class Heap {
    const static int _HEAP_SIZE = 14096;
    T v[_HEAP_SIZE];
    int pos;

    void push_up(int i) {
        int root = (i-1) / 2;
        if (v[i] < v[root]) {
            swap(v[i], v[root]);
            push_up(root);
        }
    }

    void push_down(int i) {
        int l_son = i * 2 + 1;
        int r_son = i * 2 + 2;
        int small_son = -1;
        if (l_son < pos) {
            small_son = l_son;
            if (r_son < pos && v[r_son] < v[small_son]) {
                small_son = r_son;
            }
        }

        if (small_son != -1) {
            swap(v[small_son], v[i]);
            push_down(small_son);
        }
    }

public:
    Heap() {
        pos = 0;
    }

    T top() const {
        return v[0];
    }

    void pop() {
        T val = v[0];
        v[0] = v[--pos];
        push_down(0);
    }

    void push(T a) {
        v[pos] = a;
        push_up(pos);
        ++pos;
    }

    bool empty() {
        return pos == 0;
    }
};