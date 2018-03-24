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

class STree {
public:
    class Node {
public:
        int l, r;
        LL cnt;
        Node* l_son;
        Node* r_son;
        Node(int _l, int _r) {
            l = _l;
            r = _r;
            cnt = 0;
            l_son = 0;
            r_son = 0;
        }
    };

    Node* root;

    Node* build(int l, int r) {
        Node* node = new Node(l, r);
        if (l != r) {
            int mid = (l + r) / 2;
            node->l_son = build(l, mid);
            node->r_son = build(mid + 1, r);
        } 
        return node;
    }

    STree(int l, int r) {
        root = build(l, r);
    }

    void insert(int l, int r, LL v) {
        insert(root, l, r, v);
    }

    void insert(Node* node, int l, int r, LL v) {

        int node_l = node->l;
        int node_r = node->r;
        if (l < node_l) l = node_l;
        if (r > node_r) r = node_r;
        if (r < node_l || l > node_r) return;
        if (l == node_l && r == node_r) {
            node->cnt += v;
        } else {
            insert(node->l_son, l, r, v);
            insert(node->r_son, l, r, v);
        }
    }

    LL query(int x) {
        return query(root, x);
    }

    LL query(Node* node, int x) {
        int node_l = node->l;
        int node_r = node->r;
        if (node_l == node_r && node_r == x) {
            return node->cnt;
        }

        int mid = (node_l + node_r) / 2;
        if (x <= mid) {
            LL val = query(node->l_son, x) + node->cnt;
            return val;
        }
        else {
            LL val = query(node->r_son, x) + node->cnt;
            return val;
        }
    }
};

class york {
public: void solve();
};

void york::solve() {
    int n, m, k; cin >> n >> m >> k;
    VI v(n); REP(i, n) cin >> v[i];

    VI ls(m), rs(m), ds(m); REP(i, m) cin >> ls[i] >> rs[i] >> ds[i];

    VI xs(k), ys(k); REP(i, k) cin >> xs[i] >> ys[i];

    STree q_cnt_tree(0, m-1); REP(i, k) q_cnt_tree.insert(xs[i]-1, ys[i]-1, 1LL);

    STree cnt_tree(0, n-1); REP(i, n) cnt_tree.insert(i, i, v[i]);

    REP(i, m) cnt_tree.insert(ls[i]-1, rs[i]-1, (LL)q_cnt_tree.query(i) * ds[i]);

    REP(i, n) cout << cnt_tree.query(i) << " ";
}

int main() {
  (new york()) -> solve();
  return 0;
}

