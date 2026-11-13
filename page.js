(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('model-data').textContent);
  var CTX = (DATA.meta && DATA.meta.ctx) || 8;

  /* 字元 → Qwen 參考嵌入 PCA 二維座標 [x, y] */
  var COORD = {};
  (DATA.embmap || []).forEach(function (p) { COORD[p[0]] = [p[1], p[2]]; });
  function coordText(ch) {
    var c = COORD[ch];
    return c ? c[0].toFixed(1) + ',' + c[1].toFixed(1) : '—';
  }

  function el(id) { return document.getElementById(id); }
  function set(id, txt) { var e = el(id); if (e) e.textContent = txt; }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var SYS = '你是一個簡短的中文展示模型，用繁體中文回答，避免長篇大論。';

  /* ---------- 後端與模型對應 ---------- */
  var OLLAMA_URL = 'http://localhost:11434/api/chat';
  var OLLAMA_TAGS = {
    '0.5': 'qwen2.5:0.5b',
    '3': 'qwen2.5:3b',
    '7': 'qwen2.5:7b'
  };
  var WL_CDN = 'https://esm.sh/@mlc-ai/web-llm@0.2.88/es2022/web-llm.mjs';
  var WL_MODELS = {
    '0.5': 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    '3': 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    '7': 'Qwen2.5-7B-Instruct-q4f16_1-MLC'
  };

  var state = { be: 'ollama', size: '3', ready: false, wlEngine: null, wlKey: '' };
  function activeTag() {
    return state.be === 'ollama' ? OLLAMA_TAGS[state.size] : WL_MODELS[state.size];
  }

  /* ---------- 靜態中繼資料 ---------- */
  (function fillMeta() {
    var m = DATA.meta || {};
    var qwen = m.emb_source === 'qwen2.5-7b';
    set('embSrc', qwen ? 'Qwen2.5-7B' : '本模型自訓向量');
    set('embDim', qwen ? 3584 : 32);
    var matchTxt = '';
    if (qwen && m.qwen_matched) {
      matchTxt = m.qwen_matched + ' 字直接對應';
      if (m.qwen_borrowed) matchTxt += ' · ' + m.qwen_borrowed + ' 字借用最近鄰';
    }
    set('embMatch', matchTxt);
  })();

  /* ---------- 開機閘門 ---------- */
  var boot = el('boot'), bootStatus = el('bootStatus'), bootEnter = el('bootEnter');

  function refreshStats() {
    set('statBe', state.be === 'ollama' ? 'Ollama' : 'WebGPU');
    set('statModel', activeTag());
    set('statEmb', 'Qwen 嵌入');
    var m = DATA.meta || {};
    set('statSent', m.sentences ? m.sentences + ' 句' : '—');
    set('topNote', (state.be === 'ollama' ? 'ollama' : 'webgpu') + ' / ' + activeTag());
  }

  function setBe(be) {
    state.be = be;
    el('beSeg').querySelectorAll('button').forEach(function (b) {
      b.setAttribute('aria-pressed', b.getAttribute('data-be') === be ? 'true' : 'false');
    });
    refreshStats();
  }

  function setSize(size) {
    state.size = size;
    ['bootSizeSeg', 'heroSeg'].forEach(function (id) {
      var box = el(id);
      if (!box) return;
      box.querySelectorAll('button').forEach(function (b) {
        b.setAttribute('aria-pressed', b.getAttribute('data-model') === size ? 'true' : 'false');
      });
    });
    refreshStats();
  }

  el('beSeg').addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('button') : null;
    if (b) setBe(b.getAttribute('data-be'));
  });
  ['bootSizeSeg', 'heroSeg'].forEach(function (id) {
    el(id).addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('button') : null;
      if (b) setSize(b.getAttribute('data-model'));
    });
  });

  function reopenBoot(msg) {
    if (msg) {
      bootStatus.classList.add('err');
      bootStatus.textContent = msg;
    }
    state.ready = false;
    boot.classList.remove('hide');
  }

  /* Ollama 連線檢查：服務可達，且選定模型已安裝 */
  function checkOllama() {
    var tag = OLLAMA_TAGS[state.size];
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, 6000);
    return fetch('http://localhost:11434/api/tags', { signal: ctrl.signal })
      .catch(function () {
        throw new Error('連不上 Ollama（localhost:11434）。請確認服務已啟動；若從 file:// 或外部網址開檔被 CORS 擋下，請用本機伺服器以 http://localhost 開啟，或設定 OLLAMA_ORIGINS=* 後重啟。');
      })
      .then(function (resp) {
        clearTimeout(timer);
        if (!resp.ok) throw new Error('Ollama 回應錯誤 ' + resp.status);
        return resp.json();
      })
      .then(function (data) {
        var names = (data.models || []).map(function (m) { return m.name; });
        var hit = names.some(function (n) { return n === tag || n.indexOf(tag) === 0; });
        if (!hit) throw new Error('模型 ' + tag + ' 未安裝，請先執行：ollama pull ' + tag);
      });
  }

  /* WebLLM 引擎：動態載入 WebGPU 推論，依尺寸快取 */
  function ensureWebEngine() {
    if (!('gpu' in navigator)) {
      return Promise.reject(new Error('此瀏覽器不支援 WebGPU（需 Chrome／Edge 113+），無法使用 Web 後端。'));
    }
    var model = WL_MODELS[state.size];
    if (state.wlEngine && state.wlKey === model) return Promise.resolve();
    bootStatus.textContent = '載入 WebLLM 引擎…';
    return import(WL_CDN)
      .then(function (mod) {
        bootStatus.textContent = '下載 Qwen 模型（首次約數百 MB，之後由瀏覽器快取）…';
        return mod.CreateMLCEngine(model, {
          initProgressCallback: function (p) { if (p && p.text) bootStatus.textContent = p.text; }
        });
      })
      .then(function (engine) {
        state.wlEngine = engine;
        state.wlKey = model;
      });
  }

  function bootConnect() {
    bootEnter.disabled = true;
    bootEnter.textContent = '連線中…';
    bootStatus.classList.remove('err');
    bootStatus.textContent = '連線中…';
    var pending = state.be === 'ollama' ? checkOllama() : ensureWebEngine();
    job.then(function () {
      state.ready = true;
      boot.classList.add('hide');
      refreshStats();
    }).catch(function (err) {
      bootStatus.classList.add('err');
      bootStatus.textContent = (err && err.message) ? err.message : '連線失敗';
    }).finally(function () {
      bootEnter.disabled = false;
      bootEnter.textContent = '進入';
    });
  }

  bootEnter.addEventListener('click', bootConnect);
  el('switchBe').addEventListener('click', function () { reopenBoot('重新選擇後端或尺寸。'); });

  /* ---------- 操作台 ---------- */
  var genBtn = el('gen'), tempEl = el('temp'), topkEl = el('topk'),
      genOut = el('genOut'), ctxline = el('ctxline'), probList = el('probList'),
      pplOut = el('pplOut');
  var job = null;

  tempEl.addEventListener('input', function () { el('tempVal').textContent = (+this.value).toFixed(2); });
  topkEl.addEventListener('input', function () { el('topkVal').textContent = this.value; });

  el('chips').addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('.chip') : null;
    if (!b) return;
    el('prompt').value = b.textContent;
    startJob();
  });

  genBtn.addEventListener('click', function () {
    if (job) {
      job.stop = true;
      if (job.controller) job.controller.abort();
      return;
    }
    startJob();
  });

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

  function beginJob(chars) {
    genBtn.textContent = '停止';
    genBtn.classList.add('pause');
    renderOut(chars, true);
    renderTraj(chars);
    ctxline.textContent = '上下文：' + chars.slice(-CTX).join('');
  }

  function finishJob(my) {
    my.stop = true;
    if (job !== my) return;
    job = null;
    genBtn.textContent = '生成';
    genBtn.classList.remove('pause');
    if (!my.done) { renderOut(my.chars, false); renderTraj(my.chars); my.done = true; }
  }

  function startJob() {
    if (!state.ready) { reopenBoot('請先連線 Qwen 後再生成。'); return; }
    if (job) {
      job.stop = true;
      if (job.controller) job.controller.abort();
      return;
    }
    var prompt = (el('prompt').value || '').trim() || '今天';
    var chars = Array.from(prompt);
    var temp = +tempEl.value, topk = +topkEl.value;
    if (state.be === 'ollama') ollamaJob(prompt, chars, temp, topk);
    else webJob(prompt, chars, temp, topk);
  }

  /* ---------- Ollama 串流生成 ---------- */
  function ollamaJob(prompt, chars, temp, topk) {
    var tag = OLLAMA_TAGS[state.size];
    var cap = chars.length + 120;
    var my = { stop: false, done: false, controller: new AbortController(), chars: chars };
    job = my;
    beginJob(chars);
    probList.innerHTML = '<div class="probtag"><span class="ch">' + esc(tag) +
      '</span><span class="pc">本機 Ollama</span></div>';
    pplOut.textContent = '連線 Ollama 中…';

    function fail(reason) { finishJob(my); pplOut.textContent = reason; }

    (async function () {
      var resp;
      try {
        resp = await fetch(OLLAMA_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: my.controller.signal,
          body: JSON.stringify({
            model: tag,
            messages: [
              { role: 'system', content: SYS },
              { role: 'user', content: prompt }
            ],
            stream: true,
            options: { temperature: temp, top_k: topk, num_predict: 120 }
          })
        });
      } catch (err) {
        if (my.stop) return;
        fail('連不上 Ollama：' + (err && err.message ? err.message : '未知錯誤'));
        return;
      }

      if (!resp.ok) {
        var body = '';
        try { body = await resp.text(); } catch (e) {}
        if (resp.status === 404) {
          fail('模型 ' + tag + ' 未安裝，請先執行：ollama pull ' + tag);
        } else {
          fail('Ollama 回應錯誤 ' + resp.status + (body ? '：' + body.slice(0, 120) : ''));
        }
        return;
      }

      pplOut.textContent = '生成中…';
      var reader = resp.body.getReader();
      var dec = new TextDecoder();
      var buf = '';
      try {
        while (true) {
          var rd = await reader.read();
          if (rd.done) break;
          buf += dec.decode(rd.value, { stream: true });
          var idx;
          while ((idx = buf.indexOf('\n')) >= 0) {
            var line = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 1);
            if (!line) continue;
            var obj;
            try { obj = JSON.parse(line); } catch (e) { continue; }
            if (obj.error) { fail(obj.error); return; }
            var delta = obj.message && obj.message.content ? obj.message.content : '';
            if (delta) {
              Array.from(delta).forEach(function (ch) {
                if (chars.length < cap) chars.push(ch);
              });
              renderOut(chars, true);
              renderTraj(chars);
              ctxline.textContent = '上下文：' + chars.slice(-CTX).join('');
            }
            if (obj.done) break;
          }
          if (my.stop) { reader.cancel(); break; }
        }
        finishJob(my);
      } catch (err) {
        if (my.stop) { finishJob(my); return; }
        fail('生成中斷：' + (err && err.message ? err.message : '未知錯誤'));
      }
    })();
  }

  /* ---------- WebLLM 串流生成 ---------- */
  function webJob(prompt, chars, temp, topk) {
    var cap = chars.length + 120;
    var my = { stop: false, done: false, controller: null, chars: chars };
    job = my;
    beginJob(chars);
    probList.innerHTML = '<div class="probtag"><span class="ch">Qwen</span><span class="pc">瀏覽器 WebGPU</span></div>';
    pplOut.textContent = '準備 Web 引擎…';

    function fail(reason) { finishJob(my); pplOut.textContent = reason; }

    (async function () {
      try {
        await ensureWebEngine();
        pplOut.textContent = 'Qwen 生成中…';
        var stream = await state.wlEngine.chat.completions.create({
          messages: [
            { role: 'system', content: SYS },
            { role: 'user', content: prompt }
          ],
          stream: true,
          temperature: temp,
          top_k: topk,
          max_tokens: 120
        });
        for await (var chunk of stream) {
          if (my.stop) break;
          var delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta)
            ? chunk.choices[0].delta.content : '';
          if (delta) {
            Array.from(delta).forEach(function (ch) {
              if (chars.length < cap) chars.push(ch);
            });
            renderOut(chars, true);
            renderTraj(chars);
            ctxline.textContent = '上下文：' + chars.slice(-CTX).join('');
          }
        }
        finishJob(my);
      } catch (err) {
        fail('Qwen 載入或推論失敗：' + (err && err.message ? err.message : '未知錯誤'));
      }
    })();
  }

  /* ---------- 對話模式（Qwen） ---------- */
  var chatLog = el('chatlog'), chatIn = el('chatIn'), chatSend = el('chatSend'),
      chatNote = el('chatNote');
  var chatBusy = false;

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

  function ollamaChat(text) {
    var tag = OLLAMA_TAGS[state.size];
    return fetch(OLLAMA_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: tag,
        messages: [
          { role: 'system', content: SYS },
          { role: 'user', content: text }
        ],
        stream: false,
        options: { temperature: 0.55, top_k: 3, num_predict: 120 }
      })
    }).then(function (resp) {
      if (!resp.ok) {
        return resp.text().then(function (b) {
          throw new Error('Ollama ' + resp.status + (b ? '：' + b.slice(0, 100) : ''));
        });
      }
      return resp.json();
    }).then(function (d) {
      return d.message && d.message.content ? d.message.content.trim() : '';
    });
  }

  function webChat(text) {
    return ensureWebEngine().then(function () {
      return state.wlEngine.chat.completions.create({
        messages: [
          { role: 'system', content: SYS },
          { role: 'user', content: text }
        ],
        stream: false,
        temperature: 0.55,
        top_k: 3,
        max_tokens: 120
      });
    }).then(function (r) {
      var c = r.choices && r.choices[0] && r.choices[0].message ? r.choices[0].message.content : '';
      return (c || '').trim();
    });
  }

  function sendChat() {
    if (chatBusy) return;
    if (!state.ready) { reopenBoot('請先連線 Qwen 再對話。'); return; }
    var text = (chatIn.value || '').trim();
    if (!text) return;
    chatIn.value = '';
    addMsg(text, 'user');
    chatBusy = true;
    chatSend.disabled = true;
    chatSend.textContent = '寫…';
    var pending = addMsg('…', 'bot wait');

    var reply = state.be === 'ollama' ? ollamaChat(text) : webChat(text);
    reply.then(function (out) {
      chatLog.removeChild(pending);
      if (!out) addMsg('（接不下去了，換個說法試試。）', 'bot');
      else addMsg(out, 'bot', activeTag());
      chatNote.textContent = '回答由 ' + (state.be === 'ollama' ? '本機 Ollama' : '瀏覽器 WebGPU') +
        ' 上的 ' + activeTag() + ' 產生。';
    }).catch(function (err) {
      chatLog.removeChild(pending);
      addMsg('（連線失敗：' + (err && err.message ? err.message : '未知錯誤') + '）', 'bot');
    }).finally(function () {
      chatBusy = false;
      chatSend.disabled = false;
      chatSend.textContent = '送出';
    });
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

  /* ---------- 字元座標圖 ---------- */
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
    var axisText = { color: '#8D887A', fontSize: 11, fontFamily: 'monospace' };
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
  refreshStats();
  window.addEventListener('resize', function () {
    if (embChart) embChart.resize();
    if (trajChart) trajChart.resize();
  });
})();
