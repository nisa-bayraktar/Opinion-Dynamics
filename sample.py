def update_nodes():
    global g
    nds = list(g.nodes)
    shuffle(nds)
    for i in nds:
        nbs = list(g.neighbors(i))
        if len(nbs) > 0:
            # social contagion
            total = sum(g[i][j]['weight'] for j in nbs)
            if total > 0:
                av = sum(g[i][j]['weight'] * g.nodes[j]['state'][0] for j in nbs) / total
                g.nodes[i]['state'][0] += c * (av - g.nodes[i]['state'][0]) * Dt
            for j in nbs:
                # homophily
                diff = abs(g.nodes[i]['state'][0] - g.nodes[j]['state'][0])
                g[i][j]['weight'] += h * (hth - diff) * Dt
                # novelty
                if total > 0:
                    diff = abs(av - g.nodes[j]['state'][0])
                    g[i][j]['weight'] += a * (diff - ath) * Dt
                if g[i][j]['weight'] < 0:
                    g[i][j]['weight'] = 0
        g.nodes[i]['state'][0] += normal(0, 0.1)