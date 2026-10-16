/**
 * Gold Day Trade Assistant - Frontend Application
 */

// Singleton audio context with automatic user-gesture unlocking
let sharedAudioCtx = null;

function getAudioContext() {
  if (!sharedAudioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      sharedAudioCtx = new AudioContextClass();
    }
  }
  if (sharedAudioCtx && sharedAudioCtx.state === 'suspended') {
    sharedAudioCtx.resume().catch(() => {});
  }
  return sharedAudioCtx;
}

// Unlock audio on first user click anywhere on page
document.addEventListener('click', function unlockAudioAndNotify() {
  getAudioContext();
  if (window.Notification && Notification.permission === "default") {
    Notification.requestPermission();
  }
  document.removeEventListener('click', unlockAudioAndNotify);
}, { once: true });

// Sound alert generator using Web Audio API (Pleasant 2-tone trading bell)
function playAlertChime() {
  try {
    const audioCtx = getAudioContext();
    if (!audioCtx) return;

    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }

    // Note 1: High crisp chime (E6)
    const osc1 = audioCtx.createOscillator();
    const gain1 = audioCtx.createGain();
    osc1.type = 'triangle';
    osc1.frequency.setValueAtTime(1318.51, audioCtx.currentTime); // E6
    gain1.gain.setValueAtTime(0.35, audioCtx.currentTime);
    gain1.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.8);
    osc1.connect(gain1);
    gain1.connect(audioCtx.destination);
    osc1.start();
    osc1.stop(audioCtx.currentTime + 0.8);

    // Note 2: Harmonious chime (B6) slightly delayed
    const osc2 = audioCtx.createOscillator();
    const gain2 = audioCtx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(1975.53, audioCtx.currentTime + 0.15); // B6
    gain2.gain.setValueAtTime(0.3, audioCtx.currentTime + 0.15);
    gain2.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 1.2);
    osc2.connect(gain2);
    gain2.connect(audioCtx.destination);
    osc2.start(audioCtx.currentTime + 0.15);
    osc2.stop(audioCtx.currentTime + 1.2);
  } catch (e) {
    console.warn("Web Audio API alert suppressed or error:", e);
  }
}

function testSoundAlert() {
  playAlertChime();
  if (window.Notification && Notification.permission !== "granted") {
    Notification.requestPermission();
  }
}

// TradingView Widget Setup with retry loop
function initTradingViewWidget(retries = 10) {
  const container = document.getElementById("tradingview_widget");
  if (!container) return;

  if (typeof TradingView !== "undefined") {
    container.innerHTML = "";
    try {
      new TradingView.widget({
        "autosize": true,
        "symbol": "OANDA:XAUUSD",
        "interval": "1",
        "timezone": "Asia/Bangkok",
        "theme": "dark",
        "style": "1",
        "locale": "th_TH",
        "toolbar_bg": "#111827",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "container_id": "tradingview_widget",
        "studies": [
          "MASimple@tv-basicstudies",
          "RSI@tv-basicstudies"
        ]
      });
    } catch (e) {
      console.warn("TradingView widget init error:", e);
    }
  } else if (retries > 0) {
    setTimeout(() => initTradingViewWidget(retries - 1), 500);
  } else {
    container.innerHTML = '<div class="h-full flex items-center justify-center text-gray-500 text-xs">TradingView Widget Offline / กำลังโหลด...</div>';
  }
}

// Render Analysis Data to UI
function updateUIWithAnalysis(data) {
  if (!data) return;

  document.getElementById("analysisTimestamp").textContent = data.timestamp || "--";
  document.getElementById("currentPrice").textContent = "$" + (data.current_price?.toFixed(2) || "--");
  
  const priceSourceEl = document.getElementById("priceSource");
  if (data.source && data.source.includes("Simulated")) {
    priceSourceEl.textContent = data.source;
    priceSourceEl.className = "text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-semibold";
  } else {
    priceSourceEl.textContent = data.source || "Live";
    priceSourceEl.className = "text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400";
  }

  // Signal Badge
  const badge = document.getElementById("signalBadge");
  const badgeContainer = document.getElementById("signalBadgeContainer");
  
  if (data.signal === "BUY") {
    badge.className = "px-5 py-2 rounded-2xl text-base font-extrabold tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/20";
    badge.textContent = "🟢 BUY (ซื้อ / ขึ้น)";
  } else if (data.signal === "SELL") {
    badge.className = "px-5 py-2 rounded-2xl text-base font-extrabold tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/40 shadow-lg shadow-rose-500/20";
    badge.textContent = "🔴 SELL (ขาย / ลง)";
  } else {
    badge.className = "px-5 py-2 rounded-2xl text-base font-extrabold tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/40";
    badge.textContent = "⏳ WAIT (รอจังหวะ)";
  }

  // TQF Quality Badge
  const tqfBadge = document.getElementById("tqfQualityBadge");
  if (tqfBadge) {
    if (data.quality_tier || data.quality_score != null) {
      tqfBadge.classList.remove("hidden");
      const score = data.quality_score ?? 80;
      const tier = data.quality_tier || "GOOD";
      if (tier === "EXCELLENT") {
        tqfBadge.className = "text-xs font-bold px-2.5 py-1 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5 shadow-sm shadow-emerald-500/10";
        tqfBadge.innerHTML = `<span>🛡️ คุณภาพ TQF: เกรด A (${score}/100)</span>`;
      } else if (tier === "GOOD") {
        tqfBadge.className = "text-xs font-bold px-2.5 py-1 rounded-xl bg-blue-500/20 text-blue-300 border border-blue-500/40 flex items-center gap-1.5";
        tqfBadge.innerHTML = `<span>⚡ คุณภาพ TQF: เกรด B (${score}/100)</span>`;
      } else if (tier === "CAUTION") {
        tqfBadge.className = "text-xs font-bold px-2.5 py-1 rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1.5";
        tqfBadge.innerHTML = `<span>⚠️ คุณภาพ TQF: เฝ้าระวัง (${score}/100)</span>`;
      } else {
        tqfBadge.className = "text-xs font-bold px-2.5 py-1 rounded-xl bg-rose-500/20 text-rose-300 border border-rose-500/40 flex items-center gap-1.5";
        tqfBadge.innerHTML = `<span>⛔ กรอง TQF ออก (${score}/100)</span>`;
      }
    } else {
      tqfBadge.classList.add("hidden");
    }
  }

  document.getElementById("actionText").textContent = data.action_th || "กำลังเตรียมแผน...";
  document.getElementById("entryRange").textContent = data.entry_range || "--";
  document.getElementById("confidenceVal").textContent = (data.confidence || 50) + "%";
  document.getElementById("tpLevel").textContent = `TP1: $${data.take_profit_1?.toFixed(2) || "--"}`;
  document.getElementById("tpLevel2").textContent = `TP2: $${data.take_profit_2?.toFixed(2) || "--"}`;
  document.getElementById("slLevel").textContent = `SL: $${data.stop_loss?.toFixed(2) || "--"}`;
  document.getElementById("rrRatio").textContent = `ระยะ R:R ${data.risk_reward_ratio || "1:1.5"}`;

  // IQ Option Box
  const mm = data.money_management || {};
  const directionSpan = document.getElementById("iqDirection");
  if (data.signal === "BUY") {
    directionSpan.className = "font-bold px-3 py-1 rounded bg-emerald-600 text-white";
    directionSpan.textContent = "กดปุ่มเขียว (HIGHER / BUY)";
  } else if (data.signal === "SELL") {
    directionSpan.className = "font-bold px-3 py-1 rounded bg-rose-600 text-white";
    directionSpan.textContent = "กดปุ่มแดง (LOWER / SELL)";
  } else {
    directionSpan.className = "font-bold px-3 py-1 rounded bg-amber-600 text-white";
    directionSpan.textContent = "ยังไม่กด รอสัญญาณชัดเจน";
  }

  const profitText = (mm.est_profit_tp1_usd != null) ? mm.est_profit_tp1_usd.toFixed(2) : "5.00";
  const lossText = (mm.est_loss_sl_usd != null) ? mm.est_loss_sl_usd.toFixed(2) : "2.50";
  document.getElementById("iqTpPct").textContent = `+${mm.iq_tp_slider_pct ?? 50}% (ได้กำไร: +$${profitText} USD)`;
  document.getElementById("iqSlPct").textContent = `-${mm.iq_sl_slider_pct ?? 25}% (ยอมเสีย: -$${lossText} USD)`;
  document.getElementById("iqSummaryText").textContent = mm.summary_th || "สไนเปอร์ 2:1 เป้าหมาย +$5.00 / ยอมเสีย -$2.50 (กฎ 1 ไม้ต่อวัน)";

  // Trailing Stop Card
  const step1El = document.getElementById("trailingStep1");
  const step2El = document.getElementById("trailingStep2");
  const step3El = document.getElementById("trailingStep3");
  const trailingSummaryEl = document.getElementById("trailingSummaryText");

  if (step1El) {
    step1El.textContent = mm.step1_be_trigger ? `เมื่อราคาแตะ $${mm.step1_be_trigger} ➔ เลื่อน SL ไปที่ $${data.entry_price || '--'} (กันทุน $0)` : "เมื่อกำไรแตะ +50% ➔ เลื่อน SL สู่จุดเข้า (Break-Even $0.00)";
  }
  if (step2El) {
    step2El.textContent = mm.step2_lock_trigger ? `เมื่อราคาแตะ $${mm.step2_lock_trigger} ➔ เลื่อน SL ล็อคกำไรที่ $${mm.step2_locked_sl} (+${(mm.est_step2_profit_usd || 1.5).toFixed(2)}$)` : "เมื่อกำไรแตะ +75% ➔ เลื่อน SL ล็อคกำไรที่ +$1.50 (+15%)";
  }
  if (step3El) {
    step3El.textContent = data.take_profit_1 ? `+100% ชนเป้า $${data.take_profit_1} (+${profitText}$) แล้วปิดจอทันที` : "+100% ชนเป้า +$5.00 แล้วปิดจอทันที";
  }
  if (trailingSummaryEl) {
    trailingSummaryEl.textContent = "กลยุทธ์ 2-Step Trailing Stop ช่วยแปลงไม้ที่เกือบถึง TP แต่ย้อนกลับ ให้กลายเป็นไม้ชนะล็อคกำไร +$1.50!";
  }

  // Reasons list
  const reasonsList = document.getElementById("reasonsList");
  reasonsList.innerHTML = "";
  if (data.reasons && data.reasons.length > 0) {
    data.reasons.forEach(r => {
      const li = document.createElement("li");
      li.className = "flex items-start gap-2";
      li.innerHTML = `<span class="text-amber-500 mt-0.5">•</span><span>${r}</span>`;
      reasonsList.appendChild(li);
    });
  }

  // Stats
  if (data.indicators) {
    if (document.getElementById("statEma9") && data.indicators.ema9 != null) {
      document.getElementById("statEma9").textContent = "$" + data.indicators.ema9;
    }
    if (document.getElementById("statEma21") && data.indicators.ema21 != null) {
      document.getElementById("statEma21").textContent = "$" + data.indicators.ema21;
    }
    if (document.getElementById("statEma50") && data.indicators.ema50 != null) {
      document.getElementById("statEma50").textContent = "$" + data.indicators.ema50;
    }
    document.getElementById("statAsianHigh").textContent = "$" + data.indicators.asian_high;
    document.getElementById("statAsianLow").textContent = "$" + data.indicators.asian_low;
    document.getElementById("statRsi").textContent = data.indicators.rsi;
    document.getElementById("statAtr").textContent = "$" + data.indicators.atr;
  }
}

// Render History Table
function updateHistoryTable(historyList) {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  if (!Array.isArray(historyList) || historyList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="px-4 py-4 text-center text-gray-500">ยังไม่มีบันทึกประวัติการวิเคราะห์</td></tr>`;
    return;
  }

  tbody.innerHTML = historyList.filter(Boolean).map(item => {
    let badgeClass = "bg-gray-800 text-gray-300";
    if (item.signal === "BUY") badgeClass = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
    if (item.signal === "SELL") badgeClass = "bg-rose-500/20 text-rose-400 border border-rose-500/30";
    if (item.signal === "WAIT") badgeClass = "bg-amber-500/20 text-amber-400 border border-amber-500/30";

    const reason = (item.reasons && item.reasons[0]) ? item.reasons[0].substring(0, 35) + "..." : "-";
    const estProfit = item.money_management ? `+$${item.money_management.est_profit_tp1_usd}` : "-";

    let timeDisplay = "--:--";
    if (typeof item.timestamp === "string") {
      const parts = item.timestamp.split(" ");
      timeDisplay = parts.length > 1 ? parts[1] : item.timestamp;
    } else if (item.timestamp != null) {
      timeDisplay = String(item.timestamp);
    }

    let qualityBadgeHtml = "";
    if (item.quality_tier) {
      let qClass = "bg-gray-800 text-gray-400";
      if (item.quality_tier === "EXCELLENT") qClass = "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      else if (item.quality_tier === "GOOD") qClass = "bg-blue-500/20 text-blue-300 border border-blue-500/30";
      else if (item.quality_tier === "CAUTION") qClass = "bg-amber-500/20 text-amber-300 border border-amber-500/30";
      else if (item.quality_tier === "FILTERED") qClass = "bg-rose-500/20 text-rose-300 border border-rose-500/30";
      qualityBadgeHtml = `<span class="ml-1 px-1.5 py-0.5 rounded text-[9px] font-semibold ${qClass}">${item.quality_tier}</span>`;
    }

    return `
      <tr class="hover:bg-gray-800/40 transition">
        <td class="px-4 py-3 font-mono text-gray-400">${timeDisplay}</td>
        <td class="px-4 py-3 whitespace-nowrap"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${badgeClass}">${item.signal || "--"}</span>${qualityBadgeHtml}</td>
        <td class="px-4 py-3 font-mono text-white">$${item.entry_price || "--"}</td>
        <td class="px-4 py-3 font-mono text-emerald-400">$${item.take_profit_1 || "--"}</td>
        <td class="px-4 py-3 font-mono text-rose-400">$${item.stop_loss || "--"}</td>
        <td class="px-4 py-3 font-mono font-semibold text-emerald-300">${estProfit}</td>
        <td class="px-4 py-3 text-gray-400 text-[11px]">${reason}</td>
      </tr>
    `;
  }).join("");
}

// Trigger Manual Analysis Button
async function runManualAnalysis() {
  const btn = document.getElementById("analyzeBtn");
  const spinner = document.getElementById("analyzeSpinner");
  const btnText = document.getElementById("analyzeBtnText");
  const investment = document.getElementById("inputInvestment").value || 10;
  const multiplier = document.getElementById("inputMultiplier").value || 200;

  try {
    spinner.classList.remove("hidden");
    btnText.textContent = "กำลังวิเคราะห์...";
    btn.disabled = true;

    const res = await fetch(`/api/analyze?investment=${investment}&multiplier=${multiplier}`, {
      method: "POST"
    });
    const data = await res.json();
    if (!res.ok) {
      console.warn("Manual analysis failed with status:", res.status, data);
      let errorMsg = "การวิเคราะห์ล้มเหลว ตรวจสอบค่าเงินลงทุนและ Multiplier";
      if (typeof data.detail === "string") {
        errorMsg = data.detail;
      } else if (Array.isArray(data.detail)) {
        errorMsg = data.detail.map(d => d.msg || JSON.stringify(d)).join(", ");
      } else if (typeof data.detail === "object" && data.detail !== null) {
        errorMsg = data.detail.msg || data.detail.message || JSON.stringify(data.detail);
      }
      alert(errorMsg);
      return;
    }
    updateUIWithAnalysis(data);
    fetchHistory();
    playAlertChime();
  } catch (err) {
    console.error("Manual analysis failed:", err);
    alert("ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้ กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต");
  } finally {
    spinner.classList.add("hidden");
    btnText.textContent = "⚡ วิเคราะห์ทันที";
    btn.disabled = false;
  }
}

// Fetch Status (clock, countdown, scheduler trigger)
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    document.getElementById("liveClock").textContent = data.current_time_thai || "--:--:--";
    const countdownEl = document.getElementById("countdownBadge");
    const statusLabel = document.getElementById("sessionStatusLabel");
    if (data.is_in_evening_session) {
      countdownEl.textContent = "🔴 LIVE รอบค่ำ (19:00-22:00)";
      countdownEl.className = "text-sm font-mono font-bold text-rose-400 animate-pulse";
      if (statusLabel) statusLabel.textContent = "สถานะตลาดรอบค่ำ";
    } else {
      countdownEl.textContent = data.countdown_to_session || data.countdown_to_1900 || data.countdown_to_1500 || "--:--:--";
      countdownEl.className = "text-sm font-mono font-bold text-emerald-400";
      if (statusLabel) statusLabel.textContent = "นับถอยหลังรอบค่ำ 19:00";
    }

    if (data.has_new_alert) {
      show1500Alert();
    }
  } catch (e) {
    console.error("Status fetch error:", e);
  }
}

// Fetch History
async function fetchHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    updateHistoryTable(data);
  } catch (e) {
    console.error("History fetch error:", e);
  }
}

// Alert Popup (Evening 19:00 / 1M Sniper)
function show1500Alert() {
  const banner = document.getElementById("alertBanner");
  banner.classList.remove("hidden");
  playAlertChime();

  if (window.Notification && Notification.permission === "granted") {
    new Notification("🔔 สไนเปอร์ 1M ตลาด New York / London Overlap (19:00 น.)!", {
      body: "บทวิเคราะห์ทองคำ 1M Sniper (+5$ TP / -2.5$ SL) พร้อมแล้ว กฎ 1 ไม้ต่อวัน เข้ามาดูจุดเข้าออเดอร์บนแดชบอร์ดได้เลยครับ",
      icon: "https://cryptologos.cc/logos/pax-gold-paxg-logo.png"
    });
  }

  // Refresh latest data
  fetch("/api/latest")
    .then(r => r.json())
    .then(data => {
      updateUIWithAnalysis(data);
      fetchHistory();
    });
}

function dismissBanner() {
  document.getElementById("alertBanner").classList.add("hidden");
}

// Inputs change event (auto-recalculate on change without full refetch if we have data)
document.getElementById("inputInvestment").addEventListener("change", runManualAnalysis);
document.getElementById("inputMultiplier").addEventListener("change", runManualAnalysis);

// Initialize on load
window.addEventListener("DOMContentLoaded", () => {
  initTradingViewWidget();
  
  // Fetch initial data
  fetch("/api/latest")
    .then(r => r.json())
    .then(data => updateUIWithAnalysis(data))
    .catch(console.error);

  fetchHistory();
  fetchStatus();

  // Status polling every 2.5 seconds
  setInterval(fetchStatus, 2500);
});

// ==========================================
// BACKTEST REPORT MODAL & EQUITY CURVE LOGIC
// ==========================================
let backtestCache = null;
let currentBtSize = "10";
let currentBtStrategy = "filtered"; // "filtered" or "baseline"

function switchBacktestStrategy(strat) {
  currentBtStrategy = strat;
  const btnF = document.getElementById("btStratFiltered");
  const btnB = document.getElementById("btStratBaseline");

  if (btnF && btnB) {
    if (strat === "filtered") {
      btnF.className = "px-3 py-1.5 rounded-lg text-xs font-bold transition bg-emerald-500 text-black shadow";
      btnB.className = "px-3 py-1.5 rounded-lg text-xs font-bold transition text-gray-400 hover:text-white";
    } else {
      btnB.className = "px-3 py-1.5 rounded-lg text-xs font-bold transition bg-amber-500 text-black shadow";
      btnF.className = "px-3 py-1.5 rounded-lg text-xs font-bold transition text-gray-400 hover:text-white";
    }
  }
  renderBacktestData();
}

function getActiveBacktestData() {
  if (!backtestCache) return null;
  if (currentBtStrategy === "baseline" && backtestCache.baseline) {
    return backtestCache.baseline;
  }
  if (currentBtStrategy === "filtered" && backtestCache.filtered) {
    return backtestCache.filtered;
  }
  return backtestCache;
}

async function openBacktestModal() {
  const modal = document.getElementById("backtestModal");
  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";

  if (!backtestCache) {
    try {
      const res = await fetch("/api/backtest");
      backtestCache = await res.json();
    } catch (e) {
      console.error("Failed to load backtest data:", e);
      return;
    }
  }

  // Sync size buttons UI with currentBtSize
  ["2", "5", "10"].forEach(s => {
    const btn = document.getElementById(`sizeTab${s}`);
    if (btn) {
      if (s === currentBtSize) {
        btn.className = "px-3 py-1 rounded-lg text-xs font-bold transition bg-amber-500 text-black shadow";
      } else {
        btn.className = "px-3 py-1 rounded-lg text-xs font-bold transition text-gray-400 hover:text-white";
      }
    }
  });

  switchBacktestStrategy(currentBtStrategy);
}

function closeBacktestModal() {
  document.getElementById("backtestModal").classList.add("hidden");
  document.body.style.overflow = "auto";
}

function switchBacktestSize(sz) {
  currentBtSize = sz;
  ["2", "5", "10"].forEach(s => {
    const btn = document.getElementById(`sizeTab${s}`);
    if (btn) {
      if (s === sz) {
        btn.className = "px-3 py-1 rounded-lg text-xs font-bold transition bg-amber-500 text-black shadow";
      } else {
        btn.className = "px-3 py-1 rounded-lg text-xs font-bold transition text-gray-400 hover:text-white";
      }
    }
  });

  const pnlHeader = document.getElementById("btMonthlyPnlHeader");
  if (pnlHeader) {
    pnlHeader.textContent = `กำไรสุทธิ ($${sz} ไม้)`;
  }

  renderBacktestMetrics();
  const activeData = getActiveBacktestData();
  if (activeData) {
    renderMonthlyTable(activeData.monthly_summary || []);
  }
}

function renderBacktestData() {
  const activeData = getActiveBacktestData();
  if (!activeData) return;

  // Dynamic Dataset Header
  const ds = (backtestCache && backtestCache.dataset) || activeData.dataset || {};
  const badgeEl = document.getElementById("btDatasetBadge");
  if (badgeEl && ds.total_candles) {
    badgeEl.textContent = `${ds.total_candles.toLocaleString()} แท่งเทียน (${(ds.timeframe || '1M').toUpperCase()})`;
  }
  const subtitleEl = document.getElementById("btDatasetSubtitle");
  if (subtitleEl && ds.session) {
    subtitleEl.textContent = `ข้อมูลจริงทองคำ (${ds.symbol || 'XAU/USD'}) ช่วงเวลา ${ds.session} (1-Trade-Per-Day, R:R 2:1)`;
  }

  // Dynamic Comparison Banner Cards
  if (backtestCache && backtestCache.comparison) {
    const cmp = backtestCache.comparison;
    const imp = cmp.improvement || {};
    const bStats = cmp.baseline || {};
    const fStats = cmp.filtered || {};

    const cmpSubEl = document.getElementById("btCmpSubtitle");
    if (cmpSubEl && backtestCache.filter_breakdown) {
      const bd = backtestCache.filter_breakdown;
      cmpSubEl.textContent = `คัดกรองไม้เสี่ยงทิ้ง ${bd.filtered_trades || 0} ไม้ ยกระดับ Profit Factor ขึ้นเป็น ${fStats.profit_factor || '1.21'}x`;
    }

    const nonLossEl = document.getElementById("cmpNonLossVal");
    if (nonLossEl && bStats.non_losing_rate_pct !== undefined) {
      nonLossEl.innerHTML = `${bStats.non_losing_rate_pct}% ➔ <b class="text-emerald-400">${fStats.non_losing_rate_pct}%</b>`;
    }
    const nonLossDeltaEl = document.getElementById("cmpNonLossDelta");
    if (nonLossDeltaEl && imp.non_losing_delta_pct) {
      nonLossDeltaEl.textContent = `${imp.non_losing_delta_pct} ปลอดภัยขึ้น`;
    }

    const pfEl = document.getElementById("cmpPfVal");
    if (pfEl && bStats.profit_factor !== undefined) {
      pfEl.innerHTML = `${bStats.profit_factor} ➔ <b class="text-emerald-400">${fStats.profit_factor}</b>`;
    }
    const pfDeltaEl = document.getElementById("cmpPfDelta");
    if (pfDeltaEl && imp.profit_factor_multiplier) {
      pfDeltaEl.textContent = `${imp.profit_factor_multiplier} เท่าตัว`;
    }

    const ddEl = document.getElementById("cmpDdVal");
    if (ddEl) {
      const bDd = bStats.max_drawdown_10_usd !== undefined ? bStats.max_drawdown_10_usd : (bStats.max_drawdown_5_usd || 0);
      const fDd = fStats.max_drawdown_10_usd !== undefined ? fStats.max_drawdown_10_usd : (fStats.max_drawdown_5_usd || 0);
      ddEl.innerHTML = `$${bDd} ➔ <b class="text-emerald-400">$${fDd}</b>`;
    }
    const ddDeltaEl = document.getElementById("cmpDdDelta");
    if (ddDeltaEl && imp.max_drawdown_reduction_pct) {
      ddDeltaEl.textContent = `${imp.max_drawdown_reduction_pct} ความเสี่ยงลดลง`;
    }

    const netCmpEl = document.getElementById("cmpNetVal");
    if (netCmpEl) {
      const bNet = bStats.net_pnl_10_usd !== undefined ? bStats.net_pnl_10_usd : (bStats.net_pnl_5_usd || 0);
      const fNet = fStats.net_pnl_10_usd !== undefined ? fStats.net_pnl_10_usd : (fStats.net_pnl_5_usd || 0);
      netCmpEl.innerHTML = `$${bNet >= 0 ? '+' : ''}${bNet.toFixed(2)} ➔ <b class="text-emerald-400">$${fNet >= 0 ? '+' : ''}${fNet.toFixed(2)}</b>`;
    }
    const netCmpDeltaEl = document.getElementById("cmpNetDelta");
    if (netCmpDeltaEl && imp.net_profit_delta_usd) {
      netCmpDeltaEl.textContent = `${imp.net_profit_delta_usd} USD (พลิกมีกำไร)`;
    }
  }

  const stats = activeData.overall_stats || {};
  document.getElementById("btWinRate").textContent = `${stats.win_rate_pct}%`;
  document.getElementById("btWinsCount").textContent = `${stats.wins} ไม้`;

  document.getElementById("btBeRate").textContent = `${stats.break_even_pct}%`;
  document.getElementById("btBeCount").textContent = `${stats.break_evens} ไม้ (กันทุน)`;

  document.getElementById("btLossRate").textContent = `${stats.loss_rate_pct}%`;
  document.getElementById("btLossesCount").textContent = `${stats.losses} ไม้`;

  document.getElementById("btNonLossRate").textContent = `${stats.non_losing_rate_pct}%`;
  document.getElementById("btTotalTrades").textContent = `จากทั้งหมด ${stats.total_trades} ไม้`;

  const eqSubtitleEl = document.getElementById("equityCurveSubtitle");
  if (eqSubtitleEl && stats.total_trades) {
    eqSubtitleEl.textContent = `คำนวณสะสมทีละไม้ตลอด ${stats.total_trades} การเทรด`;
  }

  renderBacktestMetrics();
  renderMonthlyTable(activeData.monthly_summary || []);
}

function renderBacktestMetrics() {
  const activeData = getActiveBacktestData();
  if (!activeData) return;
  const sizeData = (activeData.by_size && activeData.by_size[currentBtSize]) || {};
  
  const net = sizeData.total_pnl_usd || 0;
  const netEl = document.getElementById("btNetProfit");
  netEl.textContent = `${net >= 0 ? '+' : ''}$${net.toFixed(2)} USD`;
  netEl.className = `text-lg font-mono font-bold ${net >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;

  document.getElementById("btRoi").textContent = `+${sizeData.roi_pct || 0}%`;
  document.getElementById("btProfitFactor").textContent = sizeData.profit_factor || "1.0";
  document.getElementById("btMaxDd").textContent = `-$${sizeData.max_drawdown_usd || 0} USD`;

  // Draw Equity Curve
  const curve = (activeData.equity_curves && activeData.equity_curves[currentBtSize]) || [];
  drawEquityCurve(curve);
}

function drawEquityCurve(data) {
  const canvas = document.getElementById("equityCanvas");
  if (!canvas || !data || data.length === 0) return;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;

  const ctx = canvas.getContext("2d");
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  const minVal = Math.min(0, ...data);
  const maxVal = Math.max(1, ...data);
  const range = (maxVal - minVal) || 1;

  const padTop = 15;
  const padBottom = 20;
  const padLeft = 45;
  const padRight = 20;

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const getY = (val) => padTop + plotH - ((val - minVal) / range) * plotH;
  const getX = (idx) => padLeft + (idx / (data.length - 1)) * plotW;

  // Draw Zero Line
  const zeroY = getY(0);
  ctx.strokeStyle = "rgba(107, 114, 128, 0.4)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, zeroY);
  ctx.lineTo(w - padRight, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Y Axis Labels
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`$${maxVal.toFixed(1)}`, padLeft - 6, padTop + 10);
  ctx.fillText(`$0.0`, padLeft - 6, zeroY + 3);
  if (minVal < 0) {
    ctx.fillText(`$${minVal.toFixed(1)}`, padLeft - 6, h - padBottom);
  }

  // Draw Curve Area Gradient
  const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
  grad.addColorStop(0, "rgba(16, 185, 129, 0.35)");
  grad.addColorStop(1, "rgba(16, 185, 129, 0.0)");

  ctx.beginPath();
  ctx.moveTo(getX(0), zeroY);
  for (let i = 0; i < data.length; i++) {
    ctx.lineTo(getX(i), getY(data[i]));
  }
  ctx.lineTo(getX(data.length - 1), zeroY);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw Curve Line
  ctx.beginPath();
  ctx.moveTo(getX(0), getY(data[0]));
  for (let i = 1; i < data.length; i++) {
    ctx.lineTo(getX(i), getY(data[i]));
  }
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Draw Final Point Dot
  const lastX = getX(data.length - 1);
  const lastY = getY(data[data.length - 1]);
  ctx.fillStyle = "#34d399";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fill();
}

function renderMonthlyTable(monthlyList) {
  const tbody = document.getElementById("btMonthlyBody");
  if (!tbody) return;

  tbody.innerHTML = monthlyList.map(m => {
    const pnlKey = `net_pnl_${currentBtSize}_usd`;
    const pnl = m[pnlKey] !== undefined ? m[pnlKey] : (m.net_pnl_5_usd || 0);
    const isPos = pnl >= 0;
    return `
      <tr class="hover:bg-gray-800/40 transition">
        <td class="px-3 py-2 font-bold text-gray-200">${m.month}</td>
        <td class="px-3 py-2 text-gray-400">${m.trades}</td>
        <td class="px-3 py-2 text-emerald-400">${m.wins}</td>
        <td class="px-3 py-2 text-rose-400">${m.losses}</td>
        <td class="px-3 py-2 text-amber-400">${m.be}</td>
        <td class="px-3 py-2 font-bold text-emerald-300">${m.win_rate}%</td>
        <td class="px-3 py-2 text-right font-bold ${isPos ? 'text-emerald-400' : 'text-rose-400'}">
          ${isPos ? '+' : ''}$${pnl.toFixed(2)}
        </td>
      </tr>
    `;
  }).join("");
}

