#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <queue>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>
using namespace std;

#define REP(A, B) for (int A = 0; A < B; ++A)
#define FOR(A, B, C) for (int A = B; A < C; ++A)
#define SZ(A) (A.size())
#define PB push_back
#define ALL(X) (X).begin(), (X).end()
#define debug(X) cout<<"  ... "#X" : "<<(X)<<"\n" 
#define uniq(A) sort(ALL(A));(A).erase(unique((A).begin(),(A).end()),(A).end()) 
#define conv(A,B) {stringstream _3xian;_3xian<<A;_3xian>>B;} 
typedef long long LL;
typedef vector<int> VI;
typedef string STR;

const int MAX_N = 512;

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

class ppp {
public:
    int d;
    int id;
    ppp(int _d, int _id) {
        d = _d;
        id = _id;
    }

    ppp() {
        d = -1;
        id = -1;
    }

    bool operator<(const ppp& p2) const {
        return d < p2.d;
    }
};

vector<int> dijkstra_sum(int conn[][MAX_N], const VI& pts, int source) {
    vector<int> dist(MAX_N, -1);
    dist[source] = 0;
    Heap<ppp> pq;
    pq.push(ppp(0, source));

    while (!pq.empty()) {
        ppp now_pair = pq.top();
        pq.pop();
        int now_dist = now_pair.d;
        int now_id = now_pair.id;
        if (now_dist != dist[now_id]) continue;

        REP(i, SZ(pts)) {
            int next_pts = pts[i];
            int next_dist = now_dist + conn[now_id][next_pts];
            if (dist[next_pts] == -1 || next_dist < dist[next_pts]) {
                dist[next_pts] = next_dist;
                pq.push(ppp(next_dist, next_pts));
            }
        }
    }

    return dist;
};

class york {
public:
  void solve();
};

void york::solve() {
    int n; cin >> n;
    int conn[MAX_N][MAX_N] = {0};
    REP(i, n) REP(j, n) cin >> conn[i][j];

    int inverse_conn[MAX_N][MAX_N] = {0};
    REP(i, n) REP(j, n) inverse_conn[i][j] = conn[j][i];

    VI pts;
    VI remove_order(n);
    REP(i, n) cin >> remove_order[i];
    reverse(ALL(remove_order));
    vector<LL> ans_list;
    int shortest[MAX_N][MAX_N] = {-1};
    // memset(shortest, -1, sizeof(shortest));

    REP(i, n) {
        int pt = remove_order[i] - 1;
        pts.push_back(pt);
        
        vector<int> c_dist = dijkstra_sum(conn, pts, pt);
        vector<int> inv_dist = dijkstra_sum(inverse_conn, pts, pt);

        LL v = 0;
        REP(i, SZ(pts)) {
            REP(j, SZ(pts)) {
                if (i == j) continue;
                int pi = pts[i];
                int pj = pts[j];
                if (pi == pt || pj == pt) continue;

                int new_path = inv_dist[pi] + c_dist[pj];
                if (shortest[pi][pi] == -1 || new_path < shortest[pi][pj]) {
                    shortest[pi][pj] = new_path;
                }
                v += shortest[pi][pj];
            }
        }

        REP(i, SZ(pts)) {
            int pi = pts[i];
            v += c_dist[pi];
            v += inv_dist[pi];
            shortest[pt][pi] = c_dist[pi];
            shortest[pi][pt] = inv_dist[pi];
        }

        ans_list.push_back(v);
    }

    reverse(ALL(ans_list));
    REP(i, n) {
        cout << ans_list[i] << " ";
    }
}

int main() {
  (new york()) -> solve();
  return 0;
}

