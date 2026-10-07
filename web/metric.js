(function (root) {
  function emd(a, b) {
    const n = a.length,
      m = b.length,
      N = n + m + 2,
      s = N - 2,
      t = N - 1,
      g = Array.from({ length: N }, () => []);
    function edge(u, v, cap, cost) {
      const x = { v, cap, cost, rev: g[v].length },
        y = { v: u, cap: 0, cost: -cost, rev: g[u].length };
      g[u].push(x);
      g[v].push(y);
    }
    const sa = a.reduce((s, r) => s + r[0], 0),
      sb = b.reduce((s, r) => s + r[0], 0);
    a.forEach((r, i) => edge(s, i, r[0] / sa, 0));
    b.forEach((r, j) => edge(n + j, t, r[0] / sb, 0));
    a.forEach((r, i) =>
      b.forEach((q, j) =>
        edge(i, n + j, 1, Math.hypot(r[1] - q[1], r[2] - q[2], r[3] - q[3])),
      ),
    );
    let flow = 0,
      cost = 0;
    for (let iter = 0; flow < 1 - 1e-9 && iter < 100; iter++) {
      const d = Array(N).fill(Infinity),
        prev = Array(N);
      d[s] = 0;
      for (let pass = 0; pass < N - 1; pass++) {
        let changed = false;
        for (let u = 0; u < N; u++)
          for (let k = 0; k < g[u].length; k++) {
            const e = g[u][k];
            if (e.cap > 1e-10 && d[u] + e.cost < d[e.v] - 1e-10) {
              d[e.v] = d[u] + e.cost;
              prev[e.v] = [u, k];
              changed = true;
            }
          }
        if (!changed) break;
      }
      if (!prev[t]) throw Error("Palette transport failed");
      let add = 1 - flow;
      for (let v = t; v !== s; ) {
        const [u, k] = prev[v];
        add = Math.min(add, g[u][k].cap);
        v = u;
      }
      for (let v = t; v !== s; ) {
        const [u, k] = prev[v],
          e = g[u][k];
        e.cap -= add;
        g[v][e.rev].cap += add;
        v = u;
      }
      flow += add;
      cost += add * d[t];
    }
    if (flow < 1 - 1e-8) throw Error("Incomplete palette transport");
    return Math.max(0, cost);
  }
  function cosine(a, b, start, end) {
    let dot = 0,
      aa = 0,
      bb = 0;
    for (let i = start; i < end; i++) {
      dot += a[i] * b[i];
      aa += a[i] * a[i];
      bb += b[i] * b[i];
    }
    return Math.max(0, Math.min(2, 1 - dot / Math.sqrt(aa * bb)));
  }
  function distances(q, items) {
    return items.map((x) => [
      emd(q.palette, x.palette),
      cosine(q.descriptor, x.descriptor, 16, 80),
      cosine(q.descriptor, x.descriptor, 80, 144),
    ]);
  }
  function productKey(x) {
    return x.style && x.color
      ? JSON.stringify([x.style, x.color])
      : `item:${x.id}`;
  }
  function rank(q, items, ds, w, config, exclude = true, groupViews = true) {
    const total = w.reduce((a, b) => a + b, 0);
    if (!total) return [];
    const sorted = items
      .filter(
        (x) => x.id !== q.id && (!exclude || productKey(x) !== productKey(q)),
      )
      .map((x) => ({
        item: x,
        score:
          w.reduce(
            (s, v, j) => s + (v ? (v * ds[x.id][j]) / config.scales[j] : 0),
            0,
          ) / total,
      }))
      .sort(
        (a, b) =>
          Math.floor(a.score / config.tie_epsilon + 0.5) -
            Math.floor(b.score / config.tie_epsilon + 0.5) ||
          a.item.id - b.item.id,
      );
    if (!groupViews) return sorted.map((r) => ({ ...r, viewIds: [r.item.id] }));
    const groups = new Map();
    for (const r of sorted) {
      const k = productKey(r.item);
      if (groups.has(k)) groups.get(k).viewIds.push(r.item.id);
      else groups.set(k, { ...r, viewIds: [r.item.id] });
    }
    return [...groups.values()];
  }
  const api = { emd, cosine, distances, rank, productKey };
  if (typeof module !== "undefined") module.exports = api;
  else root.FashionMetric = api;
})(typeof window !== "undefined" ? window : globalThis);
