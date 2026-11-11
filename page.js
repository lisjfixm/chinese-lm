(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('model-data').textContent);
  var ITOS = DATA.itos, V = ITOS.length;
  var STOI = {};
  for (var i = 0; i < V; i++) { if (!(ITOS[i] in STOI)) STOI[ITOS[i]] = i; }
  var SOS = DATA.sos, EOS = STOI['</S>'] !== undefined ? STOI['</S>'] : 1,
      UNK = STOI['<UNK>'] !== undefined ? STOI['<UNK>'] : 2;
  var CTX = DATA.meta.ctx;
  var SH = DATA.mlp.shape;
  var EMB = SH.emb[1], HID = SH.fc1[0], IN = SH.fc1[1];

  /* 字元 → PCA 二維座標 [x, y] */
  var COORD = {};
  (DATA.embmap || []).forEach(function (p) { COORD[p[0]] = [p[1], p[2]]; });
  function coordText(ch) {
    var c = COORD[ch];
    return c ? c[0].toFixed(1) + ',' + c[1].toFixed(1) : '—';
  }

  function el(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- 把匯出的指標與維度填進靜態版面 ---------- */
  (function fillMeta() {
    var m = DATA.meta;
    function set(id, txt) { var e = el(id); if (e) e.textContent = txt; }
    set('topNote', 'char-level language model / ' + m.sentences + ' 句 / ' + V + ' 字 / ' + m.params + ' 參數');
    set('ledeCount', m.sentences + ' 句中英句子');
    set('statSent', m.sentences + ' 句');
    set('statVocab', V + ' 字');
    set('statPpl', (+m.mlp_val_ppl).toFixed(1));
    set('statCtx', CTX + ' 字');
    set('mlpDesc', '每個字先查成一條 ' + EMB + ' 維的向量，' + CTX + ' 個字接起來變成 ' + IN +
      ' 維，穿過一層 tanh 隱藏層，最後對 ' + V + ' 個字各打一個分數，softmax 成機率。');
    set('figCtxLbl', '前 ' + CTX + ' 個字');
    set('figCtxVal', 'id → ' + CTX);
    set('figEmb', V + '×' + EMB);
    set('figFc1', HID + '×' + IN);
    set('figFc2', V + '×' + HID);
    set('figVocab', V + ' 個字的機率');
    set('ngramNote', '表內為示意次數；實際頁面使用匯出的 bigram ' + m.bigram_entries + ' 組真實計數。');
    set('pplNote', '驗證集困惑度 ' + (+m.mlp_val_ppl).toFixed(1) + ' 高於訓練集 ' +
      (+m.mlp_train_ppl).toFixed(1) + '，說明模型對沒背過的組合仍不擅長預測。');
    set('embDim', EMB);
    var loss = DATA.loss || [];
    if (loss.length) {
      set('lossCap', 'cross-entropy · ' + loss.length + ' epochs');
      set('lossNote', '損失由 ' + loss[0].toFixed(3) + ' 降到 ' + loss[loss.length - 1].toFixed(3) +
        '，對應訓練困惑度約 ' + (+m.mlp_train_ppl).toFixed(1) + '。');
    }
  })();

  /* ---------- int8 反量化 ---------- */
  function deq(o) {
    var bin = atob(o.data), n = bin.length;
    var u8 = new Uint8Array(n);
    for (var i = 0; i < n; i++) u8[i] = bin.charCodeAt(i) & 255;
    var i8 = new Int8Array(u8.buffer, 0, n);
    var out = new Float32Array(n);
    for (var i = 0; i < n; i++) out[i] = i8[i] * o.scale;
    return out;
  }

  var EW = deq(DATA.mlp.emb), W1 = deq(DATA.mlp.fc1w), B1 = deq(DATA.mlp.fc1b),
      W2 = deq(DATA.mlp.fc2w), B2 = deq(DATA.mlp.fc2b);

  /* ---------- MLP 前向（回傳 586 維 logits） ---------- */
  var hin = new Float32Array(IN), hh = new Float32Array(HID), logits = new Float32Array(V);

  function mlpForward(ids) {
    var k, j, h, v;
    for (k = 0; k < CTX; k++) {
      var base = ids[k] * EMB;
      for (j = 0; j < EMB; j++) hin[k * EMB + j] = EW[base + j];
    }
    for (h = 0; h < HID; h++) {
      var s = B1[h], row = h * IN;
      for (j = 0; j < IN; j++) s += W1[row + j] * hin[j];
      hh[h] = Math.tanh(s);
    }
    for (v = 0; v < V; v++) {
      var a = B2[v], r2 = v * HID;
      for (h = 0; h < HID; h++) a += W2[r2 + h] * hh[h];
      logits[v] = a;
    }
    return logits;
  }

  /* ---------- n-gram 查表 ---------- */
  var NG_K = 0.1;
  var uni = new Float32Array(V);
  var uk = DATA.ngram.unigram;
  for (var ukey in uk) uni[+ukey] = uk[ukey];
  var uniTot = DATA.ngram.total_uni;
  var p1 = new Float32Array(V);
  for (var v1 = 0; v1 < V; v1++) p1[v1] = (uni[v1] + NG_K) / (uniTot + NG_K * V);

  var BIG = {};
  function prep(table, dst) {
    for (var key in table) {
      var arr = table[key], tot = 0, i;
      for (i = 0; i < arr.length; i++) tot += arr[i][1];
      var den = tot + NG_K * V, probs = new Float32Array(arr.length);
      for (i = 0; i < arr.length; i++) probs[i] = (arr[i][1] + NG_K) / den;
      dst[key] = { ids: arr, p: probs };
    }
  }
  prep(DATA.ngram.bigram, BIG);

  var ngp = new Float32Array(V);
  function ngForward(hist) {
    var L = hist.length, a = hist[L - 1];
    var e2 = BIG[String(a)], i, v;
    for (v = 0; v < V; v++) ngp[v] = 0.2 * p1[v];
    if (e2) { for (i = 0; i < e2.ids.length; i++) ngp[e2.ids[i][0]] += 0.8 * e2.p[i]; }
    else { for (v = 0; v < V; v++) ngp[v] += 0.8 * p1[v]; }
    return ngp;
  }

  /* ---------- 溫度 + 取樣 ---------- */
  var probBuf = new Float32Array(V), idxarr = [];

  function softmaxT(src, temp, isLogit) {
    var mx = -Infinity, v, s;
    for (v = 0; v < V; v++) {
      s = isLogit ? src[v] / temp
                  : (src[v] > 0 ? Math.log(src[v]) / temp : -Infinity);
      probBuf[v] = s;
      if (s > mx) mx = s;
    }
    if (!isFinite(mx)) mx = 0;
    var sum = 0, e;
    for (v = 0; v < V; v++) { e = Math.exp(probBuf[v] - mx); probBuf[v] = e; sum += e; }
    for (v = 0; v < V; v++) probBuf[v] /= sum;
    return probBuf;
  }

  function applyTopK(p, topk) {
    if (topk >= V) return p;
    idxarr.length = 0;
    var v;
    for (v = 0; v < V; v++) idxarr.push(v);
    idxarr.sort(function (x, y) { return p[y] - p[x]; });
    var kth = p[idxarr[Math.min(topk, V) - 1]], sum = 0;
    for (v = 0; v < V; v++) { if (p[v] < kth) { p[v] = 0; } sum += p[v]; }
    if (sum > 0) for (v = 0; v < V; v++) p[v] /= sum;
    return p;
  }

  function sampleP(p) {
    var r = Math.random(), acc = 0, v;
    for (v = 0; v < V; v++) {
      if (p[v] === 0) continue;
      acc += p[v];
      if (r < acc) return v;
    }
    return V - 1;
  }

  function topList(p, n) {
    var arr = [], v;
    for (v = 0; v < V; v++) arr.push([v, p[v]]);
    arr.sort(function (x, y) { return y[1] - x[1]; });
    return arr.slice(0, n);
  }

  function enc(text) {
    var out = [], chars = Array.from(text), i, id;
    for (i = 0; i < chars.length; i++) {
      id = STOI[chars[i]];
      out.push(id === undefined ? UNK : id);
    }
    return out;
  }

  function padHist() {
    var h = [], i;
    for (i = 0; i < CTX; i++) h.push(SOS);
    return h;
  }

  function lastCtx(hist) {
    var out = [], L = hist.length, i;
    for (i = CTX; i > 0; i--) out.push(hist[L - i]);
    return out;
  }

  function distOf(model, hist, temp) {
    return model === 'mlp'
      ? softmaxT(mlpForward(lastCtx(hist)), temp, true)
      : softmaxT(ngForward(hist), temp, false);
  }

  /* ---------- 首屏字盤 ---------- */
  var band = el('band'), cells = band.children, traceAlts = el('traceAlts');
  var heroHist = null, heroPos = 0, heroTimer = null, heroLive = true;

  function clearBand() {
    for (var i = 0; i < cells.length; i++) {
      var g = cells[i].querySelector('.gl');
      var p = cells[i].querySelector('.pos');
      if (g) g.textContent = '·';
      if (p) p.innerHTML = '<span class="pn">' + (i + 1) + '</span>';
      cells[i].classList.remove('filled', 'pop');
      cells[i].classList.add('empty');
    }
  }
  function paintCell(i, ch) {
    if (i >= cells.length) return;
    var c = cells[i], g = c.querySelector('.gl'), p = c.querySelector('.pos');
    if (g) { g.textContent = ch; } else { c.textContent = ch; }
    if (p) p.innerHTML = '<span class="pn">' + (i + 1) + '</span><span class="pc">' + coordText(ch) + '</span>';
    c.classList.remove('empty');
    c.classList.add('filled', 'pop');
    setTimeout(function () { c.classList.remove('pop'); }, 320);
  }
  function labelOf(id) {
    if (id === EOS) return '終';
    if (id === UNK) return '?';
    return ITOS[id];
  }
  function showAlts(p) {
    var top = topList(p, 3), parts = [];
    for (var i = 0; i < top.length; i++) {
      parts.push('<span>' + esc(labelOf(top[i][0])) + '<i>' + Math.round(top[i][1] * 100) + '%</i></span>');
    }
    traceAlts.innerHTML = parts.join('');
  }

  function heroStep() {
    heroTimer = null;
    if (!heroLive || slowMotion) return;
    if (!heroHist) {
      heroHist = padHist().concat(enc('今'));
      clearBand();
      paintCell(0, '今');
      heroPos = 1;
    }
    var p = distOf('mlp', heroHist, 0.6);
    applyTopK(p, 3);
    var b = sampleP(p);
    showAlts(p);
    if (b === EOS || heroPos >= cells.length) {
      heroHist = null;
      traceAlts.textContent = '換一句…';
      heroTimer = setTimeout(heroStep, 1400);
      return;
    }
    heroHist.push(b);
    paintCell(heroPos, ITOS[b]);
    heroPos++;
    heroTimer = setTimeout(heroStep, 620);
  }

  function resumeHero(delay) {
    if (slowMotion || !heroLive || heroTimer) return;
    heroTimer = setTimeout(heroStep, delay || 200);
  }

  var slowMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (slowMotion) {
    ['今', '天', '天', '氣', '很', '好'].forEach(function (c, i) { paintCell(i, c); });
    traceAlts.textContent = '示意畫面（已依系統設定關閉自動播放）';
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      heroLive = entries[0].isIntersecting;
      if (heroLive) resumeHero(200);
    }, { threshold: 0.12 }).observe(band);
  } else {
    resumeHero(400);
  }

  /* ---------- 操作台 ---------- */
  var genBtn = el('gen'), tempEl = el('temp'), topkEl = el('topk'),
      genOut = el('genOut'), ctxline = el('ctxline'), probList = el('probList'),
      pplOut = el('pplOut');
  var curModel = 'mlp', job = null;

  el('modelSeg').addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('button') : null;
    if (!b) return;
    var kids = this.querySelectorAll('button');
    for (var i = 0; i < kids.length; i++) {
      kids[i].setAttribute('aria-pressed', kids[i] === b ? 'true' : 'false');
    }
    curModel = b.getAttribute('data-model');
    probList.innerHTML = '<div class="prow"><div class="ch">—</div><div class="track"></div><div class="pc">—</div></div>';
    pplOut.textContent = '—';
  });

  el('chips').addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('.chip') : null;
    if (!b) return;
    el('prompt').value = b.textContent;
    startJob();
  });

  tempEl.addEventListener('input', function () { el('tempVal').textContent = (+this.value).toFixed(2); });
  topkEl.addEventListener('input', function () { el('topkVal').textContent = this.value; });

  genBtn.addEventListener('click', function () {
    if (job) { job.stop = true; return; }
    startJob();
  });

  function renderDist(p, chosen) {
    var top = topList(p, 8), html = '', max = top[0][1] || 1, i;
    for (i = 0; i < top.length; i++) {
      var id = top[i][0], w = (top[i][1] / max) * 100, isC = id === chosen;
      html += '<div class="prow"><div class="ch"' + (isC ? ' style="color:var(--cin)"' : '') + '>' +
        esc(labelOf(id)) + '</div><div class="track"><div class="fill" style="width:' + w.toFixed(1) + '%' +
        (isC ? '' : ';background:#6B6558') + '"></div></div><div class="pc">' +
        (top[i][1] * 100).toFixed(1) + '%</div></div>';
    }
    probList.innerHTML = html;
  }

  function renderOut(chars, streaming) {
    var html = '', i, n = chars.length;
    for (i = 0; i < n; i++) {
      var cur = (i === n - 1 && streaming);
      html += '<span class="tk' + (cur ? ' cur' : '') + '"><i>' + coordText(chars[i]) + '</i><b>' +
        esc(chars[i]) + '</b></span>';
    }
    if (streaming) html += '<span class="caret"></span>';
    genOut.innerHTML = html;
  }

  /* ---------- 生成軌跡座標圖 ---------- */
  var trajChart = null;
  function trajPoints(chars) {
    var pts = [];
    chars.forEach(function (ch) {
      var c = COORD[ch];
      if (c) pts.push([c[0], c[1], ch]);
    });
    return pts;
  }
  function renderTraj(chars) {
    var box = el('trajMap');
    if (!box) return;
    var pts = trajPoints(chars);
    var cnt = el('trajCount');
    if (cnt) cnt.textContent = pts.length + ' 字';
    if (typeof echarts === 'undefined') {
      box.innerHTML = '<p style="color:#8D887A;font-size:12.5px;line-height:1.9;padding-top:12px">軌跡圖需要 ECharts 連線載入。</p>';
      return;
    }
    if (!trajChart) trajChart = echarts.init(box);
    var lineData = pts.map(function (p) { return [p[0], p[1]]; });
    trajChart.setOption({
      backgroundColor: 'transparent',
      grid: { left: 18, right: 18, top: 18, bottom: 28 },
      tooltip: {
        trigger: 'item',
        backgroundColor: '#1C1D15', borderColor: 'rgba(231,225,210,.2)',
        textStyle: { color: '#E7E1D2', fontSize: 11 },
        formatter: function (p) {
          var d = p.data;
          return esc(d[2]) + '　(' + d[0].toFixed(2) + ', ' + d[1].toFixed(2) + ')';
        }
      },
      xAxis: {
        type: 'value', min: -1.05, max: 1.05,
        axisLine: { lineStyle: { color: 'rgba(231,225,210,.18)' } },
        axisTick: { show: false },
        axisLabel: { color: '#8D887A', fontSize: 11, fontFamily: 'monospace' },
        splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
      },
      yAxis: {
        type: 'value', min: -1.05, max: 1.05,
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: '#8D887A', fontSize: 11, fontFamily: 'monospace' },
        splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
      },
      series: [
        {
          type: 'line', data: lineData, showSymbol: false, zlevel: 1,
          lineStyle: { color: '#C0452E', width: 1.5 }
        },
        {
          type: 'scatter', data: pts, symbolSize: 7, zlevel: 2,
          itemStyle: { color: '#E7E1D2' },
          label: {
            show: true, position: 'top', color: '#C9C2B2', fontSize: 10,
            formatter: function (p) { return p.data[2]; }
          }
        }
      ]
    });
  }

  function startJob() {
    if (job) job.stop = true;
    var model = curModel;
    var prompt = (el('prompt').value || '').trim() || '今天';
    var chars = Array.from(prompt);
    var hist = padHist().concat(enc(prompt));
    var temp = +tempEl.value, topk = +topkEl.value;
    var cap = chars.length + 26;
    var my = { stop: false, logp: 0, n: 0 };
    job = my;

    genBtn.textContent = '停止';
    genBtn.classList.add('pause');
    renderOut(chars, true);
    renderTraj(chars);
    ctxline.textContent = '上下文：' + (chars.slice(-CTX).join('') || '—');
    pplOut.textContent = '計算中…';
    probList.innerHTML = '';

    function finish() {
      my.stop = true;
      if (job !== my) return;
      job = null;
      genBtn.textContent = '生成';
      genBtn.classList.remove('pause');
      if (!my.done) {
        renderOut(chars, false);
        renderTraj(chars);
        my.done = true;
      }
      if (my.n > 0) {
        pplOut.textContent = '本次生成困惑度 ' + Math.exp(-my.logp / my.n).toFixed(1);
      }
    }

    function tick() {
      if (my.stop) { finish(); return; }
      if (chars.length >= cap) { finish(); return; }
      var p = distOf(model, hist, temp);
      applyTopK(p, topk);
      var chosen = sampleP(p);
      if (chosen === EOS) { renderDist(p, chosen); finish(); return; }
      renderDist(p, chosen);
      hist.push(chosen);
      chars.push(ITOS[chosen]);
      my.logp += Math.log(Math.max(p[chosen], 1e-12));
      my.n += 1;
      pplOut.textContent = '本次生成困惑度 ' + Math.exp(-my.logp / my.n).toFixed(1);
      ctxline.textContent = '上下文：' + chars.slice(-CTX).join('');
      renderOut(chars, true);
      renderTraj(chars);
      setTimeout(tick, 62);
    }
    setTimeout(tick, 90);
  }

  /* ---------- 兩引擎對比 ---------- */
  var vRun = el('vRun');
  vRun.addEventListener('click', function () {
    var prompt = (el('prompt').value || '').trim() || '今天';
    var temp = +tempEl.value, topk = +topkEl.value;
    var base = padHist().concat(enc(prompt));
    var target = 22;

    function runOne(kind) {
      var hist = base.slice(), chars = Array.from(prompt), logp = 0, n = 0, c, p;
      while (chars.length < prompt.length + target) {
        p = distOf(kind, hist, temp);
        applyTopK(p, topk);
        c = sampleP(p);
        if (c === EOS) break;
        hist.push(c);
        chars.push(ITOS[c]);
        logp += Math.log(Math.max(p[c], 1e-12));
        n++;
      }
      el(kind === 'mlp' ? 'vMlp' : 'vNg').textContent = chars.join('');
      el(kind === 'mlp' ? 'vMlpMeta' : 'vNgMeta').textContent =
        (kind === 'mlp' ? 'MLP' : 'n-gram') + '　共 ' + n + ' 字　生成困惑度 ' +
        Math.exp(-logp / Math.max(n, 1)).toFixed(1);
    }

    vRun.textContent = '生成中…';
    el('vMlp').textContent = '…';
    el('vNg').textContent = '…';
    el('vMlpMeta').textContent = '—';
    el('vNgMeta').textContent = '—';
    setTimeout(function () {
      runOne('mlp');
      runOne('ngram');
      vRun.textContent = '兩邊同時生成';
    }, 80);
  });


  /* ---------- 位置實驗表 ---------- */
  (function renderProbe() {
    var rows = (DATA.meta.probe || []), html = '', i, j;
    for (i = 0; i < rows.length; i++) {
      html += '<div class="row"><span>' + esc(rows[i].ctx) + '</span>';
      for (j = 0; j < 3; j++) {
        var it = rows[i].top[j] || ['—', 0];
        html += '<span><b>' + esc(it[0]) + '</b><em>' + Number(it[1]).toFixed(1) + '%</em></span>';
      }
      html += '</div>';
    }
    var box = document.getElementById('probeRows');
    if (box) box.innerHTML = html;
  })();

  /* ---------- 對話模式 ---------- */
  var chatLog = el('chatlog'), chatIn = el('chatIn'), chatSend = el('chatSend'),
      chatNote = el('chatNote');
  var chatBusy = false;

  function stripLead(t) {
    return String(t).replace(/^[—－\-，,、。；;：:\s]+/, '');
  }

  var DLG = DATA.meta.dialogs || [], DLGG = [];
  (function prepDialog() {
    for (var i = 0; i < DLG.length; i++) DLGG.push({ q: DLG[i][0], a: DLG[i][1], g: grams(DLG[i][0]) });
  })();

  function grams(str) {
    var a = Array.from(str), out = {}, i;
    for (i = 0; i < a.length - 1; i++) out[a[i] + a[i + 1]] = 1;
    return out;
  }

  function dice(a, b) {
    var hit = 0, k, na = 0, nb = 0;
    for (k in a) { na++; if (k in b) hit++; }
    for (k in b) nb++;
    return (na + nb) ? (2 * hit) / (na + nb) : 0;
  }

  function nearest(text) {
    var g = grams(text), best = null, i, sc;
    for (i = 0; i < DLGG.length; i++) {
      sc = dice(g, DLGG[i].g);
      if (!best || sc > best.score) best = { score: sc, q: DLGG[i].q, a: DLGG[i].a };
    }
    return best;
  }

  function continueFrom(promptText, temp, topk, trials) {
    var promptChars = Array.from(promptText), best = null, s, t;
    for (s = 0; s < trials; s++) {
      var hist = padHist().concat(enc(promptText));
      var chars = promptChars.slice(), logp = 0, n = 0, out = '';
      for (t = 0; t < 20; t++) {
        var p = distOf('mlp', hist, temp);
        applyTopK(p, topk);
        var c = sampleP(p);
        if (c === EOS) break;
        hist.push(c);
        chars.push(ITOS[c]);
        logp += Math.log(Math.max(p[c], 1e-12));
        n++;
        out += ITOS[c];
        if (n >= 2 && /[。？！]/.test(out.charAt(out.length - 1))) break;
      }
      var score = logp / Math.max(n, 1);
      if (n > 0 && (!best || score > best.score)) best = { text: out, score: score, n: n };
    }
    return best;
  }

  function genReply(text) {
    var n = nearest(text);
    if (n && n.score >= 0.3) {
      return { text: n.a, note: '取自語料「' + n.q + '」，相似度 ' + n.score.toFixed(2) };
    }
    var r = continueFrom(text + (/([。？！?!.])$/.test(text) ? '' : '。') + '—', 0.45, 2, 3);
    return {
      text: r ? stripLead(r.text) : '',
      note: '沒有相近的問法，改成讓模型直接接話（所以容易接得怪）'
    };
  }

  function addMsg(text, who, meta) {
    var d = document.createElement('div');
    d.className = 'msg ' + who;
    d.textContent = text;
    if (meta) {
      var sm = document.createElement('small');
      sm.textContent = meta;
      d.appendChild(sm);
    }
    chatLog.appendChild(d);
    chatLog.scrollTop = chatLog.scrollHeight;
    return d;
  }

  function sendChat() {
    if (chatBusy) return;
    var text = (chatIn.value || '').trim();
    if (!text) return;
    chatIn.value = '';
    addMsg(text, 'user');
    chatBusy = true;
    chatSend.disabled = true;
    chatSend.textContent = '寫…';
    var pending = addMsg('…', 'bot wait');

    setTimeout(function () {
      var r = genReply(text);
      chatLog.removeChild(pending);
      if (!r || !r.text) {
        addMsg('（接不下去了，換個說法試試。）', 'bot');
      } else {
        addMsg(r.text, 'bot', r.note);
      }
      chatNote.textContent = '回覆取自語料裡最接近的那組問答；沒有相近問法時才讓模型自己接話。';
      chatBusy = false;
      chatSend.disabled = false;
      chatSend.textContent = '送出';
    }, 30);
  }

  chatSend.addEventListener('click', sendChat);
  chatIn.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); sendChat(); }
  });
  el('chatChips').addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('.chip') : null;
    if (!b) return;
    chatIn.value = b.textContent;
    sendChat();
  });

  /* ---------- 圖表 ---------- */
  function drawCharts() {
    if (typeof echarts === 'undefined') {
      el('chartLoss').innerHTML = '<p style="color:#8D887A;font-size:12.5px;line-height:1.9;padding-top:12px">損失 ' + DATA.loss[0].toFixed(2) + ' → ' + DATA.loss[DATA.loss.length - 1].toFixed(2) + '（' + DATA.loss.length + ' 輪）。</p>';
      el('chartPpl').innerHTML = '<p style="color:#8D887A;font-size:12.5px;line-height:1.9;padding-top:12px">2-gram ' + DATA.meta.ngram_bigram_ppl + '　3-gram ' + DATA.meta.ngram_trigram_ppl + '<br>MLP 訓練 ' + DATA.meta.mlp_train_ppl + '　MLP 驗證 ' + DATA.meta.mlp_val_ppl + '</p>';
      return;
    }
    var axisText = { color: '#8D887A', fontSize: 11, fontFamily: 'monospace' };
    var tip = {
      trigger: 'axis',
      backgroundColor: '#1C1D15',
      borderColor: 'rgba(231,225,210,.2)',
      textStyle: { color: '#E7E1D2', fontSize: 11 }
    };

    var loss = echarts.init(el('chartLoss'));
    loss.setOption({
      backgroundColor: 'transparent',
      grid: { left: 36, right: 12, top: 18, bottom: 26 },
      tooltip: tip,
      xAxis: {
        type: 'category',
        data: DATA.loss.map(function (_, i) { return i + 1; }),
        axisLine: { lineStyle: { color: 'rgba(231,225,210,.18)' } },
        axisTick: { show: false },
        axisLabel: { color: '#8D887A', fontSize: 11, fontFamily: 'monospace', interval: 4 }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: axisText,
        splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
      },
      series: [{
        type: 'line',
        data: DATA.loss.map(function (x) { return +x.toFixed(4); }),
        symbol: 'circle',
        symbolSize: 3,
        lineStyle: { color: '#C0452E', width: 1.6 },
        itemStyle: { color: '#C0452E' },
        areaStyle: { color: 'rgba(192,69,46,.10)' }
      }]
    });

    var ppl = echarts.init(el('chartPpl'));
    ppl.setOption({
      backgroundColor: 'transparent',
      grid: { left: 36, right: 12, top: 18, bottom: 26 },
      tooltip: tip,
      xAxis: {
        type: 'category',
        data: ['2-gram', '3-gram', 'MLP 訓練', 'MLP 驗證'],
        axisLine: { lineStyle: { color: 'rgba(231,225,210,.18)' } },
        axisTick: { show: false },
        axisLabel: { color: '#8D887A', fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: axisText,
        splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
      },
      series: [{
        type: 'bar',
        barWidth: '34%',
        data: [+DATA.meta.ngram_bigram_ppl, +DATA.meta.ngram_trigram_ppl, +DATA.meta.mlp_train_ppl, +DATA.meta.mlp_val_ppl],
        itemStyle: {
          color: function (p) { return p.dataIndex === 3 ? '#C0452E' : '#84A196'; }
        },
        label: {
          show: true, position: 'top', color: '#C9C2B2',
          fontSize: 11, fontFamily: 'monospace',
          formatter: function (p) { return Number(p.value).toFixed(1); }
        }
      }]
    });

    /* ---------- 字向量座標圖 ---------- */
    var EMB_STYLE = {
      zh:    { color: '#E7E1D2', label: '中文字' },
      en:    { color: '#84A196', label: '英文字母' },
      digit: { color: '#C0452E', label: '數字' },
      punct: { color: '#8D887A', label: '標點' },
      space: { color: '#5E5A50', label: '空格' },
      sp:    { color: '#5E5A50', label: '特殊符號' }
    };
    var embChart = null;

    function embSeries(filter) {
      var byType = {}, hits = [];
      (DATA.embmap || []).forEach(function (pt) {
        var ch = pt[0], x = pt[1], y = pt[2], t = pt[3];
        var hit = !!filter && ch.indexOf(filter) >= 0;
        if (hit) { hits.push([x, y, ch]); }
        else { (byType[t] = byType[t] || []).push([x, y, ch]); }
      });
      var series = Object.keys(EMB_STYLE).map(function (t) {
        return {
          name: EMB_STYLE[t].label, type: 'scatter', data: byType[t] || [],
          symbolSize: 5, itemStyle: { color: EMB_STYLE[t].color, opacity: 0.62 }
        };
      });
      if (hits.length) {
        series.push({
          name: '命中', type: 'scatter', data: hits, symbolSize: 10, zlevel: 2,
          itemStyle: { color: '#C0452E', opacity: 0.95 }
        });
      }
      return series;
    }

    function drawEmbMap() {
      var box = el('embMap');
      if (!box || !DATA.embmap) return;
      if (typeof echarts === 'undefined') {
        box.innerHTML = '<p style="color:#8D887A;font-size:12.5px;line-height:1.9;padding-top:12px">字向量地圖需要 ECharts 連線載入；離線環境下可改看本機匯出的 模型.txt。</p>';
        return;
      }
      embChart = echarts.init(box);
      embChart.setOption({
        backgroundColor: 'transparent',
        grid: { left: 18, right: 18, top: 18, bottom: 28 },
        tooltip: {
          trigger: 'item',
          backgroundColor: '#1C1D15', borderColor: 'rgba(231,225,210,.2)',
          textStyle: { color: '#E7E1D2', fontSize: 11 },
          formatter: function (p) {
            return esc(p.data[2]) + '　(' + p.data[0].toFixed(2) + ', ' + p.data[1].toFixed(2) + ')';
          }
        },
        xAxis: {
          type: 'value', min: -1.05, max: 1.05,
          axisLine: { lineStyle: { color: 'rgba(231,225,210,.18)' } },
          axisTick: { show: false }, axisLabel: axisText,
          splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
        },
        yAxis: {
          type: 'value', min: -1.05, max: 1.05,
          axisLine: { show: false }, axisTick: { show: false },
          axisLabel: axisText,
          splitLine: { lineStyle: { color: 'rgba(231,225,210,.06)' } }
        },
        series: embSeries('')
      });
      embChart.on('click', function (p) {
        var ch = p.data[2];
        if (ch && ch[0] !== '<') {
          el('prompt').value = ch;
          el('console').scrollIntoView({ behavior: 'smooth' });
        }
      });
      var lg = el('embLegend');
      if (lg) {
        lg.innerHTML = Object.keys(EMB_STYLE).map(function (t) {
          return '<span><i style="background:' + EMB_STYLE[t].color + '"></i>' + EMB_STYLE[t].label + '</span>';
        }).join('');
      }
      var search = el('embSearch');
      if (search) {
        search.addEventListener('input', function () {
          if (embChart) embChart.setOption({ series: embSeries(this.value.trim()) });
        });
      }
    }

    drawEmbMap();
    window.addEventListener('resize', function () { loss.resize(); ppl.resize(); if (embChart) embChart.resize(); if (trajChart) trajChart.resize(); });
  }

  if (document.readyState === 'complete') { drawCharts(); }
  else { window.addEventListener('load', drawCharts); }
})();
