vector<LL> dijkstra_sum(int conn[][MAX_N], const VI& pts, int source) {
    vector<LL> dist(MAX_N);
    REP(i, MAX_N) dist[i] = -1;
    dist[source] = 0;
    priority_queue<ppp> pq;
    pq.push(ppp(0LL, source));

    while (!pq.empty()) {
        ppp now_pair = pq.top();
        pq.pop();
        LL now_dist = now_pair.d;
        int now_id = now_pair.id;
        if (now_dist != dist[now_id]) continue;

        REP(i, SZ(pts)) {
            int next_pts = pts[i];
            LL next_dist = now_dist + conn[now_id][next_pts];
            if (dist[next_pts] == -1 || next_dist < dist[next_pts]) {
                dist[next_pts] = next_dist;
                pq.push(ppp(next_dist, next_pts));
            }
        }
    }

    return dist;
};