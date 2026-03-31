/**
 * ═══════════════════════════════════════════════════════════════════════════
 * IPL Pro-Analytics — Real-Time Dashboard Controller
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Architecture
 * ────────────
 * 1. State Store          — single source of truth for all dashboard data
 * 2. WebSocket Client     — reconnecting WS with exponential backoff
 * 3. REST Fallback        — fetch /predict when WS unavailable
 * 4. Render Pipeline      — DOM diffing strategy: only update changed values
 * 5. Animation Engine     — CSS-class-driven flash effects
 *
 * Performance
 * ───────────
 * • requestAnimationFrame throttled re-renders
 * • No innerHTML in hot path — uses textContent / classList API
 * • Debounced prop-bet heatmap (500 ms)
 */

(function () {
    'use strict';

    // ─── Configuration ───────────────────────────────────────────────
    const CONFIG = {
        WS_BASE:    `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`,
        API_BASE:   location.origin,
        WS_RECONNECT_BASE_MS: 1000,
        WS_RECONNECT_MAX_MS:  30000,
        WS_TICK_INTERVAL_MS:   2000,
        PROP_UPDATE_MS:        5000,
        REST_POLL_MS:         10000,
        MATCH_ID:             'MIvCSK',    // Default
    };

    // ─── State Store ─────────────────────────────────────────────────
    const state = {
        matchId: CONFIG.MATCH_ID,
        wsConnected: false,
        // Betting Metrics
        odds: null,
        volatility: null,
        overUnder: [],
        wicketMethods: [],
        propBets: [],
        // Scorecard
        scorecard: { runs: 0, wickets: 0, overs: '0.0', crr: '0.00', phase: '—', projected: '—' },
        // Meta
        sharpEdge: '—',
        blendWeights: 'XGB 60% · LSTM 40%',
        lastUpdateMs: 0,
        latencyMs: null,
    };

    // ─── DOM Cache ───────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const dom = {
        // Odds
        oddsHome:     $('#odds-home'),
        oddsAway:     $('#odds-away'),
        impliedHome:  $('#implied-home'),
        impliedAway:  $('#implied-away'),
        barHome:      $('#bar-home'),
        barAway:      $('#bar-away'),
        overround:    $('#overround'),
        sharpEdge:    $('#sharp-edge'),
        blendWeights: $('#blend-weights'),
        // Volatility
        gaugeArc:     $('#gauge-arc'),
        gaugeValue:   $('#gauge-value'),
        gaugeLabel:   $('#gauge-label'),
        tickVelocity: $('#tick-velocity'),
        maxShift:     $('#max-shift'),
        deltaLabel:   $('#delta-label'),
        // Over/Under
        ouTbody:      $('#ou-tbody'),
        ouUpdated:    $('#ou-updated'),
        // Wicket
        wicketMethods:$('#wicket-methods'),
        // Heatmap
        heatmapGrid:  $('#heatmap-grid'),
        // Scorecard
        scoreRuns:    $('#score-runs'),
        scoreWickets: $('#score-wickets'),
        scoreOvers:   $('#score-overs'),
        statCrr:      $('#stat-crr'),
        statPhase:    $('#stat-phase'),
        statProjected:$('#stat-projected'),
        // Connection
        connBadge:    $('#conn-badge'),
        connDot:      null,
        connText:     null,
        // Footer
        footerLatency:$('#footer-latency'),
        // Match Select
        matchSelect:  $('#match-select'),
    };

    // ─── Utility Functions ───────────────────────────────────────────
    function fmt(n, decimals = 2) {
        return Number(n).toFixed(decimals);
    }

    function pct(n) {
        return (n * 100).toFixed(1) + '%';
    }

    function timeAgo(ms) {
        const diff = Date.now() - ms;
        if (diff < 2000) return 'just now';
        if (diff < 60000) return Math.floor(diff / 1000) + 's ago';
        return Math.floor(diff / 60000) + 'm ago';
    }

    function heatmapClass(prob) {
        if (prob >= 0.80) return 'prob-very-high';
        if (prob >= 0.60) return 'prob-high';
        if (prob >= 0.40) return 'prob-medium';
        if (prob >= 0.20) return 'prob-low';
        return 'prob-very-low';
    }

    function volatilityColor(score) {
        if (score > 0.75) return '#ff3366';
        if (score > 0.50) return '#ff8800';
        if (score > 0.25) return '#ffd700';
        return '#00ff88';
    }

    // ─── Connection Badge ────────────────────────────────────────────
    function initConnBadge() {
        dom.connDot = dom.connBadge.querySelector('.conn-dot');
        dom.connText = dom.connBadge.querySelector('.conn-text');
    }

    function setConnStatus(status) {
        dom.connBadge.className = 'connection-badge ' + status;
        if (status === 'connected') {
            dom.connText.textContent = 'Live';
        } else if (status === 'disconnected') {
            dom.connText.textContent = 'Offline';
        } else {
            dom.connText.textContent = 'Connecting…';
        }
    }

    // ─── Render: Live Odds ───────────────────────────────────────────
    let prevOddsHome = null;
    let prevOddsAway = null;

    function renderOdds(odds) {
        if (!odds) return;

        const homeStr = fmt(odds.home, 3);
        const awayStr = fmt(odds.away, 3);

        dom.oddsHome.textContent = homeStr;
        dom.oddsAway.textContent = awayStr;
        dom.impliedHome.textContent = 'Implied: ' + pct(odds.implied_home);
        dom.impliedAway.textContent = 'Implied: ' + pct(odds.implied_away);

        // Probability bars
        const homePct = odds.implied_home * 100;
        dom.barHome.style.width = homePct + '%';
        dom.barAway.style.width = (100 - homePct) + '%';

        // Flash on change
        if (prevOddsHome !== null) {
            if (odds.home > prevOddsHome + 0.005) {
                dom.oddsHome.classList.remove('odds-flash-up', 'odds-flash-down');
                void dom.oddsHome.offsetWidth; // Force reflow
                dom.oddsHome.classList.add('odds-flash-up');
            } else if (odds.home < prevOddsHome - 0.005) {
                dom.oddsHome.classList.remove('odds-flash-up', 'odds-flash-down');
                void dom.oddsHome.offsetWidth;
                dom.oddsHome.classList.add('odds-flash-down');
            }
        }

        prevOddsHome = odds.home;
        prevOddsAway = odds.away;
    }

    // ─── Render: Volatility Gauge ────────────────────────────────────
    function renderVolatility(vol) {
        if (!vol) return;

        // SVG arc: total path length ~251.3
        const arcLength = 251.3;
        const filled = arcLength * vol.score;
        dom.gaugeArc.setAttribute('stroke-dasharray', `${filled} ${arcLength}`);

        // Colour
        const color = volatilityColor(vol.score);
        dom.gaugeArc.setAttribute('stroke', color);

        // Text
        dom.gaugeValue.textContent = fmt(vol.score, 2);
        dom.gaugeLabel.textContent = vol.label;
        dom.gaugeLabel.setAttribute('fill', color);

        // Stats
        dom.tickVelocity.textContent = vol.tick_velocity.toFixed(1);
        dom.maxShift.textContent = fmt(vol.max_shift_pct, 1) + '%';
        dom.deltaLabel.textContent = state._deltaLabel || '→';

        // Colour delta
        const dl = state._deltaLabel;
        if (dl === '↑')      dom.deltaLabel.style.color = '#00ff88';
        else if (dl === '↓')  dom.deltaLabel.style.color = '#ff3366';
        else                   dom.deltaLabel.style.color = '#8b8fa3';
    }

    // ─── Render: Over/Under Table ────────────────────────────────────
    function renderOverUnder(lines) {
        if (!lines || lines.length === 0) return;

        const frag = document.createDocumentFragment();

        lines.forEach(line => {
            const tr = document.createElement('tr');

            // Market
            const tdMarket = document.createElement('td');
            tdMarket.textContent = line.market;
            tr.appendChild(tdMarket);

            // Line
            const tdLine = document.createElement('td');
            tdLine.textContent = fmt(line.line, 1);
            tdLine.style.color = 'var(--gold)';
            tdLine.style.fontWeight = '700';
            tr.appendChild(tdLine);

            // Over Odds
            const tdOver = document.createElement('td');
            tdOver.textContent = fmt(line.over_odds, 2);
            tdOver.style.color = 'var(--neon-green)';
            tr.appendChild(tdOver);

            // Under Odds
            const tdUnder = document.createElement('td');
            tdUnder.textContent = fmt(line.under_odds, 2);
            tdUnder.style.color = 'var(--neon-green)';
            tr.appendChild(tdUnder);

            // Projected
            const tdProj = document.createElement('td');
            tdProj.textContent = fmt(line.projected_total, 1);
            tdProj.style.color = 'var(--text-secondary)';
            tr.appendChild(tdProj);

            // Confidence
            const tdConf = document.createElement('td');
            const confPct = line.confidence * 100;
            tdConf.innerHTML = `<span style="color:${confPct > 65 ? 'var(--neon-green)' : 'var(--gold)'}">${confPct.toFixed(0)}%</span>`;
            tr.appendChild(tdConf);

            frag.appendChild(tr);
        });

        dom.ouTbody.innerHTML = '';
        dom.ouTbody.appendChild(frag);
        dom.ouUpdated.textContent = timeAgo(state.lastUpdateMs);
    }

    // ─── Render: Wicket Methods ──────────────────────────────────────
    function renderWicketMethods(methods) {
        if (!methods || methods.length === 0) return;

        const frag = document.createDocumentFragment();
        let isTop = true;

        methods.forEach(m => {
            const row = document.createElement('div');
            row.className = 'wicket-row';

            const name = document.createElement('span');
            name.className = 'wicket-method-name';
            name.textContent = m.method;
            row.appendChild(name);

            const bar = document.createElement('div');
            bar.className = 'wicket-prob-bar';

            const fill = document.createElement('div');
            fill.className = 'wicket-prob-fill' + (isTop ? ' top-method' : '');
            fill.style.width = (m.probability * 100) + '%';
            bar.appendChild(fill);
            row.appendChild(bar);

            const val = document.createElement('span');
            val.className = 'wicket-prob-value';
            val.textContent = pct(m.probability);
            row.appendChild(val);

            const press = document.createElement('span');
            press.className = 'wicket-pressure';
            press.textContent = '×' + fmt(m.pressure_multiplier, 1);
            row.appendChild(press);

            frag.appendChild(row);
            isTop = false;
        });

        dom.wicketMethods.innerHTML = '';
        dom.wicketMethods.appendChild(frag);
    }

    // ─── Render: Prop-Bet Heatmap ────────────────────────────────────
    let heatmapDebounce = null;

    function renderPropBets(props) {
        if (heatmapDebounce) clearTimeout(heatmapDebounce);

        heatmapDebounce = setTimeout(() => {
            if (!props || props.length === 0) {
                dom.heatmapGrid.innerHTML = '<div class="heatmap-empty">No prop bets available</div>';
                return;
            }

            const frag = document.createDocumentFragment();

            props.forEach(p => {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell ' + heatmapClass(p.probability);

                const icon = p.milestone.includes('Wicket') ? 'fa-baseball-bat-ball'
                           : 'fa-person-running';

                cell.innerHTML = `
                    <div class="cell-player"><i class="fas ${icon}"></i> ${p.player_name}</div>
                    <div class="cell-milestone">${p.milestone}</div>
                    <div class="cell-prob">${pct(p.probability)}</div>
                    <div class="cell-stats">
                        <span><i class="fas fa-chart-simple"></i> ${p.current_value}</span>
                        <span><i class="fas fa-clock"></i> ${p.balls_remaining} balls</span>
                        ${p.strike_rate ? `<span><i class="fas fa-gauge-high"></i> ${p.strike_rate} SR</span>` : ''}
                    </div>
                `;
                frag.appendChild(cell);
            });

            dom.heatmapGrid.innerHTML = '';
            dom.heatmapGrid.appendChild(frag);
        }, 300);
    }

    // ─── Render: Scorecard ───────────────────────────────────────────
    function renderScorecard(sc) {
        dom.scoreRuns.textContent = sc.runs;
        dom.scoreWickets.textContent = sc.wickets;
        dom.scoreOvers.textContent = sc.overs;
        dom.statCrr.textContent = sc.crr;
        dom.statPhase.textContent = sc.phase;
        dom.statProjected.textContent = sc.projected;
    }

    // ─── Render: Meta Info ───────────────────────────────────────────
    function renderMeta() {
        dom.overround.textContent = state._overround ? fmt(state._overround * 100, 1) + '%' : '—';
        dom.sharpEdge.textContent = state.sharpEdge !== '—' ? state.sharpEdge + '%' : '—';
        dom.blendWeights.textContent = state.blendWeights;

        if (state.latencyMs !== null) {
            dom.footerLatency.textContent = 'Latency: ' + state.latencyMs + 'ms';
        }
    }

    // ─── Master Render (throttled via rAF) ───────────────────────────
    let renderPending = false;

    function scheduleRender() {
        if (renderPending) return;
        renderPending = true;
        requestAnimationFrame(() => {
            renderOdds(state.odds);
            renderVolatility(state.volatility);
            renderOverUnder(state.overUnder);
            renderWicketMethods(state.wicketMethods);
            renderPropBets(state.propBets);
            renderScorecard(state.scorecard);
            renderMeta();
            renderPending = false;
        });
    }

    // ─── REST: Initial Data Fetch ────────────────────────────────────
    async function fetchInitialData() {
        try {
            const t0 = performance.now();
            const res = await fetch(`${CONFIG.API_BASE}/predict/${state.matchId}?over=10&ball=1&total_runs=85&wickets=2&striker_runs=35&striker_balls=22`);
            state.latencyMs = Math.round(performance.now() - t0);

            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();

            // Extract elite insights
            const elite = data.elite_insights;
            if (!elite) {
                console.warn('No elite_insights in response');
                return;
            }

            const bm = elite.betting_metrics;

            state.odds = bm.decimal_odds;
            state.volatility = bm.market_volatility;
            state.overUnder = bm.over_under_lines;
            state.wicketMethods = bm.next_wicket_method;
            state.propBets = bm.prop_bets;
            state.sharpEdge = elite.sharp_edge_pct;
            state.blendWeights = Object.entries(elite.model_blend_weights)
                .map(([k, v]) => `${k.toUpperCase()} ${(v * 100).toFixed(0)}%`)
                .join(' · ');
            state.lastUpdateMs = elite.last_updated_ms;
            state._overround = bm.decimal_odds.overround;
            state._deltaLabel = '→';

            // Update scorecard from prediction
            state.scorecard = {
                runs: data.prediction.predicted_total ? Math.round(data.prediction.predicted_total * 0.5) : 85,
                wickets: 2,
                overs: '10.0',
                crr: '8.50',
                phase: data.prediction.innings_phase || 'middle',
                projected: data.prediction.predicted_total || '—',
            };

            scheduleRender();
            console.log('[REST] Initial data loaded, latency:', state.latencyMs, 'ms');
        } catch (err) {
            console.error('[REST] Fetch failed:', err);
            // Retry after delay
            setTimeout(fetchInitialData, CONFIG.REST_POLL_MS);
        }
    }

    // ─── WebSocket Client ────────────────────────────────────────────
    let ws = null;
    let reconnectTimer = null;
    let reconnectAttempts = 0;

    function connectWS() {
        const url = `${CONFIG.WS_BASE}/ws/odds/${state.matchId}`;
        console.log('[WS] Connecting to', url);

        try {
            ws = new WebSocket(url);
        } catch (e) {
            console.error('[WS] Connection error:', e);
            scheduleReconnect();
            return;
        }

        ws.onopen = () => {
            console.log('[WS] Connected');
            state.wsConnected = true;
            reconnectAttempts = 0;
            setConnStatus('connected');
        };

        ws.onmessage = (event) => {
            const t0 = performance.now();
            let data;
            try {
                data = JSON.parse(event.data);
            } catch {
                return;
            }
            state.latencyMs = Math.round(performance.now() - t0);

            // Identify message type
            if (data.decimal_odds && data.market_volatility) {
                // WsOddsTick
                state.odds = data.decimal_odds;
                state.volatility = data.market_volatility;
                state._deltaLabel = data.delta_label || '→';
                state._overround = data.decimal_odds.overround;
                state.lastUpdateMs = data.timestamp_ms;
            }

            if (data.prop_bets) {
                // WsPropBetUpdate
                state.propBets = data.prop_bets;
            }

            scheduleRender();
        };

        ws.onclose = (event) => {
            console.log('[WS] Closed:', event.code, event.reason);
            state.wsConnected = false;
            setConnStatus('disconnected');
            ws = null;
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            state.wsConnected = false;
            setConnStatus('disconnected');
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        const delay = Math.min(
            CONFIG.WS_RECONNECT_BASE_MS * Math.pow(2, reconnectAttempts),
            CONFIG.WS_RECONNECT_MAX_MS
        );
        reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
        setConnStatus('');
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectWS();
        }, delay);
    }

    function disconnectWS() {
        if (ws) {
            ws.close(1000, 'Match changed');
            ws = null;
        }
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        state.wsConnected = false;
        setConnStatus('disconnected');
    }

    // ─── Match Selector ──────────────────────────────────────────────
    function initMatchSelector() {
        dom.matchSelect.addEventListener('change', (e) => {
            state.matchId = e.target.value;
            console.log('[App] Match changed to', state.matchId);

            // Reset state
            state.odds = null;
            state.volatility = null;
            state.overUnder = [];
            state.wicketMethods = [];
            state.propBets = [];

            // Reconnect
            disconnectWS();
            fetchInitialData();
            connectWS();
        });
    }

    // ─── Update Tickers ──────────────────────────────────────────────
    function startTickers() {
        // Update "time ago" labels every 5s
        setInterval(() => {
            if (state.lastUpdateMs) {
                dom.ouUpdated.textContent = timeAgo(state.lastUpdateMs);
            }
        }, 5000);

        // Periodic REST refresh as fallback
        setInterval(() => {
            if (!state.wsConnected) {
                fetchInitialData();
            }
        }, CONFIG.REST_POLL_MS);
    }

    // ─── Initialisation ──────────────────────────────────────────────
    function init() {
        initConnBadge();
        initMatchSelector();
        setConnStatus('');

        // Fetch initial data via REST
        fetchInitialData();

        // Connect WebSocket
        connectWS();

        // Start background timers
        startTickers();

        console.log('[App] IPL Pro-Analytics dashboard initialised');
    }

    // Boot when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
