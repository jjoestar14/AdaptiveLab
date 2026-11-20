/**
 * AdaptiveLab - Main Frontend Application Logic (static/js/main.js)
 */

let appState = {
  currentImageId: "sample1_uneven_light",
  currentProcessResult: null,
  activeView: "slider",
  activeTab: "tab-process",
  splitViewer: null,
  debounceTimer: null,
  currentExperimentId: null
};

// ==========================================
// TOAST NOTIFICATION SYSTEM
// ==========================================
function showToast(message, type = "info", duration = 4000) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${message}</span>
    <button style="background:none; border:none; cursor:pointer; color:inherit; font-size:1.1rem; margin-left:0.5rem;" onclick="this.parentElement.remove()">&times;</button>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    if (toast.parentElement) toast.remove();
  }, duration);
}

// ==========================================
// THEME SWITCHER
// ==========================================
function initTheme() {
  const themeBtn = document.getElementById("theme-toggle-btn");
  const savedTheme = localStorage.getItem("adaptivelab_theme") || "auto";

  function applyTheme(theme) {
    if (theme === "auto") {
      document.documentElement.removeAttribute("data-theme");
      themeBtn.innerHTML = "🌓 Tema: Sistem";
    } else if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
      themeBtn.innerHTML = "🌙 Tema: Gelap";
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      themeBtn.innerHTML = "☀️ Tema: Terang";
    }
  }

  applyTheme(savedTheme);

  themeBtn.addEventListener("click", () => {
    const current = localStorage.getItem("adaptivelab_theme") || "auto";
    let next = "dark";
    if (current === "dark") next = "light";
    else if (current === "light") next = "auto";
    else next = "dark";

    localStorage.setItem("adaptivelab_theme", next);
    applyTheme(next);
    // Refresh histogram jika sedang ditampilkan
    if (appState.currentProcessResult && appState.currentProcessResult.histogram) {
      renderHistogram("histogram-canvas", appState.currentProcessResult.histogram);
    }
  });
}

// ==========================================
// TABS NAVIGATION
// ==========================================
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
      appState.activeTab = targetId;

      // Tindakan spesifik saat tab dibuka
      if (targetId === "tab-compare") {
        runComparison();
      } else if (targetId === "tab-eval") {
        updateEvaluationView();
      }
    });
  });
}

// ==========================================
// PARAMETERS & CONTROLS BINDING
// ==========================================
function bindSliderAndNumber(sliderId, numberId, badgeId = null) {
  const slider = document.getElementById(sliderId);
  const number = document.getElementById(numberId);
  const badge = badgeId ? document.getElementById(badgeId) : null;

  if (!slider || !number) return;

  const update = (val, triggerProcess = true) => {
    slider.value = val;
    number.value = val;
    if (badge) badge.textContent = val;
    if (triggerProcess) triggerAutoProcess();
  };

  slider.addEventListener("input", (e) => update(e.target.value));
  number.addEventListener("change", (e) => {
    let val = parseInt(e.target.value, 10);
    // Pastikan ganjil jika block_size
    if (sliderId.includes("block")) {
      if (val % 2 === 0) val += 1;
      val = Math.max(3, Math.min(101, val));
    }
    update(val);
  });
}

function initAccordions() {
  document.querySelectorAll(".accordion-header").forEach((header) => {
    header.addEventListener("click", () => {
      const parent = header.parentElement;
      parent.classList.toggle("open");
    });
  });
}

function getParametersPayload() {
  const bs = parseInt(document.getElementById("param-block-size").value, 10) || 21;
  const c = parseInt(document.getElementById("param-c").value, 10) || 5;
  const engine = document.getElementById("param-engine").value || "opencv";
  const fgIsBlack = document.getElementById("param-fg-color").value === "black";

  // Pra-proses
  const gbActive = document.getElementById("toggle-gaussian-blur").checked;
  const gbKsize = parseInt(document.getElementById("param-gb-ksize").value, 10) || 3;
  const gbSigma = parseFloat(document.getElementById("param-gb-sigma").value) || 0.0;

  const mbActive = document.getElementById("toggle-median-blur").checked;
  const mbKsize = parseInt(document.getElementById("param-mb-ksize").value, 10) || 3;

  const claheActive = document.getElementById("toggle-clahe").checked;
  const claheClip = parseFloat(document.getElementById("param-clahe-clip").value) || 2.0;
  const claheTile = parseInt(document.getElementById("param-clahe-tile").value, 10) || 8;

  // Pasca-proses
  const morphOp = document.getElementById("param-morph-op").value || "none";
  const morphKsize = parseInt(document.getElementById("param-morph-ksize").value, 10) || 3;
  const morphIter = parseInt(document.getElementById("param-morph-iter").value, 10) || 1;
  const morphShape = document.getElementById("param-morph-shape").value || "rect";

  const removeSmall = document.getElementById("toggle-remove-small").checked;
  const minArea = parseInt(document.getElementById("param-min-area").value, 10) || 15;
  const removeTarget = document.getElementById("param-remove-target").value || "both";

  return {
    image_id: appState.currentImageId,
    block_size: bs % 2 === 0 ? bs + 1 : bs,
    c: c,
    engine: engine,
    foreground_is_black: fgIsBlack,
    preprocess: {
      gaussian_blur: gbActive,
      gaussian_ksize: gbKsize % 2 === 0 ? gbKsize + 1 : gbKsize,
      gaussian_sigma: gbSigma,
      median_blur: mbActive,
      median_ksize: mbKsize % 2 === 0 ? mbKsize + 1 : mbKsize,
      clahe: claheActive,
      clahe_clip_limit: claheClip,
      clahe_tile_size: claheTile
    },
    postprocess: {
      morph_op: morphOp,
      morph_ksize: morphKsize,
      morph_iter: morphIter,
      morph_shape: morphShape,
      remove_small: removeSmall,
      min_area: minArea,
      remove_target: removeTarget
    }
  };
}

function triggerAutoProcess() {
  clearTimeout(appState.debounceTimer);
  const loading = document.getElementById("loading-status");
  if (loading) loading.classList.add("active");

  appState.debounceTimer = setTimeout(async () => {
    try {
      const payload = getParametersPayload();
      const res = await API.processImage(payload);
      appState.currentProcessResult = res;
      renderProcessResults(res);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      if (loading) loading.classList.remove("active");
    }
  }, 250);
}

// ==========================================
// RENDER PROCESS RESULTS
// ==========================================
function renderProcessResults(data) {
  // Update statistik
  const s = data.statistics;
  document.getElementById("stat-resolution").textContent = s.resolution;
  document.getElementById("stat-black-pct").textContent = `${s.black_percentage}%`;
  document.getElementById("stat-white-pct").textContent = `${s.white_percentage}%`;
  document.getElementById("stat-components").textContent = s.connected_components_text_black.toLocaleString();
  document.getElementById("stat-time").textContent = `${s.execution_time_ms} ms`;

  // Update pipeline bar status
  const p = data.pipeline_stages;
  document.getElementById("pipe-gb").classList.toggle("active", p.gaussian_blur);
  document.getElementById("pipe-mb").classList.toggle("active", p.median_blur);
  document.getElementById("pipe-clahe").classList.toggle("active", p.clahe);
  document.getElementById("pipe-morph").classList.toggle("active", p.morphology);
  document.getElementById("pipe-clean").classList.toggle("active", p.remove_small);

  // Update Histogram
  if (data.histogram) {
    renderHistogram("histogram-canvas", data.histogram);
    const hs = data.histogram.stats;
    document.getElementById("hist-mean").textContent = hs.mean;
    document.getElementById("hist-std").textContent = hs.std;
    document.getElementById("hist-minmax").textContent = `${hs.min} / ${hs.max}`;
  }

  // Update Image Views
  updateMainImageView();

  // Update Ground Truth indicator badge
  const gtBadge = document.getElementById("gt-status-badge");
  if (gtBadge) {
    if (data.has_ground_truth) {
      gtBadge.textContent = "Ground Truth: Tersedia";
      gtBadge.className = "badge-tag";
      gtBadge.style.backgroundColor = "var(--color-success-light)";
      gtBadge.style.color = "var(--color-success)";
    } else {
      gtBadge.textContent = "Ground Truth: Belum Ada";
      gtBadge.className = "badge-tag";
      gtBadge.style.backgroundColor = "var(--bg-subtle)";
      gtBadge.style.color = "var(--text-muted)";
    }
  }

  // Simpan report_id untuk tombol download laporan
  if (data.report_id) {
    const dlBtn = document.getElementById("btn-download-report");
    if (dlBtn) {
      dlBtn.onclick = () => {
        window.open(`/api/export/${data.report_id}?type=csv`, "_blank");
      };
    }
  }
}

function updateMainImageView() {
  if (!appState.currentProcessResult) return;
  const imgs = appState.currentProcessResult.images;
  const mode = appState.activeView;

  const sliderElem = document.getElementById("split-slider");
  const singleElem = document.getElementById("single-view-img");

  if (mode === "slider") {
    sliderElem.style.display = "block";
    singleElem.style.display = "none";
    appState.splitViewer.setImages(imgs.original, imgs.binary_final);
  } else {
    sliderElem.style.display = "none";
    singleElem.style.display = "block";

    if (mode === "binary_final") singleElem.src = imgs.binary_final;
    else if (mode === "threshold_map") singleElem.src = imgs.threshold_map;
    else if (mode === "grayscale") singleElem.src = imgs.grayscale;
    else if (mode === "preprocessed") singleElem.src = imgs.preprocessed;
    else if (mode === "binary_raw") singleElem.src = imgs.binary_raw;
    else if (mode === "original") singleElem.src = imgs.original;
  }
}

// ==========================================
// VIEW SWITCH BUTTONS
// ==========================================
function initViewButtons() {
  const btns = document.querySelectorAll(".view-btn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appState.activeView = btn.getAttribute("data-view");
      updateMainImageView();
    });
  });

  // Tombol Unduh Citra Biner
  const dlBinaryBtn = document.getElementById("btn-download-binary");
  if (dlBinaryBtn) {
    dlBinaryBtn.addEventListener("click", () => {
      if (appState.currentProcessResult && appState.currentProcessResult.images) {
        const link = document.createElement("a");
        link.download = `adaptivelab_biner_${appState.currentImageId}.png`;
        link.href = appState.currentProcessResult.images.binary_final;
        link.click();
      }
    });
  }
}

// ==========================================
// SAMPLE CARDS & UPLOAD
// ==========================================
async function loadSamples() {
  try {
    const samples = await API.getSamples();
    const container = document.getElementById("samples-container");
    if (!container) return;
    container.innerHTML = "";

    samples.forEach((s) => {
      const card = document.createElement("div");
      card.className = `sample-card ${s.id === appState.currentImageId ? "active" : ""}`;
      card.innerHTML = `
        <img src="${s.thumbnail_url}" alt="${s.name}">
        <div class="sample-card-title">${s.name}</div>
      `;
      card.addEventListener("click", () => {
        document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        appState.currentImageId = s.id;
        showToast(`Memuat sampel: ${s.name}`, "info");
        triggerAutoProcess();
      });
      container.appendChild(card);
    });
  } catch (err) {
    showToast(err.message, "error");
  }
}

function initUpload() {
  const dropzone = document.getElementById("main-dropzone");
  const fileInput = document.getElementById("main-file-input");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await handleUploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      await handleUploadFile(fileInput.files[0]);
    }
  });
}

async function handleUploadFile(file) {
  try {
    showToast("Mengunggah citra...", "info");
    const res = await API.uploadImage(file);
    appState.currentImageId = res.image_id;
    document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));
    showToast(`Berhasil mengunggah: ${res.filename} (${res.resolution})`, "success");
    triggerAutoProcess();
  } catch (err) {
    showToast(err.message, "error");
  }
}

// ==========================================
// TAB BANDINGKAN (COMPARE)
// ==========================================
async function runComparison() {
  const container = document.getElementById("comparison-results");
  if (!container) return;

  container.innerHTML = `
    <div style="grid-column: 1/-1; text-align: center; padding: 2rem;">
      <span class="spinner"></span>
      <p style="margin-top:0.5rem; color:var(--text-muted);">Menjalankan perbandingan 4 metode...</p>
    </div>
  `;

  try {
    const gThresh = parseInt(document.getElementById("cmp-global-thresh").value, 10) || 127;
    const bs = parseInt(document.getElementById("param-block-size").value, 10) || 21;
    const c = parseInt(document.getElementById("param-c").value, 10) || 5;

    const methods = await API.compareMethods({
      image_id: appState.currentImageId,
      global_thresh: gThresh,
      block_size: bs % 2 === 0 ? bs + 1 : bs,
      c: c
    });

    container.innerHTML = "";
    Object.keys(methods).forEach((key) => {
      const m = methods[key];
      const card = document.createElement("div");
      card.className = "comparison-card";
      card.innerHTML = `
        <div class="comparison-card-header">
          <h3>${m.name}</h3>
          <span class="badge-tag">${m.time_ms} ms</span>
        </div>
        <img src="${m.image_url}" alt="${m.name}">
        <div class="comparison-card-body">
          <p style="font-size:0.8rem; margin-bottom:0.4rem; color:var(--text-muted);">${m.description}</p>
          <div class="comparison-meta">
            <span>Hitam: <b>${m.black_percent}%</b></span>
            <span>Putih: <b>${m.white_percent}%</b></span>
          </div>
          <button class="btn btn-secondary btn-sm" style="width:100%; margin-top:0.5rem;" onclick="downloadImage('${m.image_url}', '${key}')">
            Unduh PNG
          </button>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div style="grid-column: 1/-1; color: var(--color-danger);">${err.message}</div>`;
    showToast(err.message, "error");
  }
}

function downloadImage(dataUrl, name) {
  const a = document.createElement("a");
  a.download = `adaptivelab_${name}.png`;
  a.href = dataUrl;
  a.click();
}

// ==========================================
// TAB EKSPERIMEN (EXPERIMENT GRID SEARCH)
// ==========================================
function initExperiment() {
  const btnRun = document.getElementById("btn-run-experiment");
  if (!btnRun) return;

  btnRun.addEventListener("click", async () => {
    const rawBs = document.getElementById("exp-blocks").value;
    const rawC = document.getElementById("exp-c").value;
    const applyPipeline = document.getElementById("exp-apply-pipeline").checked;

    btnRun.disabled = true;
    btnRun.innerHTML = `<span class="spinner" style="width:16px; height:16px;"></span> Menjalankan...`;

    try {
      let prepCfg = null;
      let postCfg = null;
      if (applyPipeline) {
        const fullPayload = getParametersPayload();
        prepCfg = fullPayload.preprocess;
        postCfg = fullPayload.postprocess;
      }

      const res = await API.runExperiment({
        image_id: appState.currentImageId,
        block_sizes: rawBs,
        c_values: rawC,
        engine: document.getElementById("param-engine").value || "opencv",
        preprocess: prepCfg,
        postprocess: postCfg
      });

      appState.currentExperimentId = res.experiment_id;
      renderExperimentResults(res);
      showToast(`Eksperimen selesai (${res.total_combinations} kombinasi).`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btnRun.disabled = false;
      btnRun.innerHTML = `Mulai Eksperimen Grid`;
    }
  });

  // Export Buttons
  document.getElementById("btn-exp-csv").addEventListener("click", () => {
    if (!appState.currentExperimentId) {
      showToast("Jalankan eksperimen terlebih dahulu.", "warning");
      return;
    }
    window.open(`/api/export/${appState.currentExperimentId}?type=csv`, "_blank");
  });

  document.getElementById("btn-exp-zip").addEventListener("click", () => {
    if (!appState.currentExperimentId) {
      showToast("Jalankan eksperimen terlebih dahulu.", "warning");
      return;
    }
    window.open(`/api/export/${appState.currentExperimentId}?type=zip`, "_blank");
  });
}

function renderExperimentResults(data) {
  const tableHead = document.getElementById("exp-table-head");
  const tableBody = document.getElementById("exp-table-body");
  const thumbGrid = document.getElementById("exp-thumb-grid");

  if (!tableHead || !tableBody || !thumbGrid) return;

  const hasGt = data.has_ground_truth;

  // Header Table
  tableHead.innerHTML = `
    <tr>
      <th>No</th>
      <th>Block Size (N)</th>
      <th>Konstanta (C)</th>
      <th>% Hitam</th>
      <th>% Putih</th>
      <th>Waktu (ms)</th>
      ${hasGt ? `
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
        <th>Accuracy</th>
        <th>PSNR (dB)</th>
      ` : ""}
    </tr>
  `;

  // Rows Table
  tableBody.innerHTML = "";
  thumbGrid.innerHTML = "";

  data.results.forEach((row, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><b>${row.block_size}</b></td>
      <td><b>${row.c}</b></td>
      <td>${row.black_percent}%</td>
      <td>${row.white_percent}%</td>
      <td>${row.time_ms} ms</td>
      ${hasGt ? `
        <td>${row.precision_percent}%</td>
        <td>${row.recall_percent}%</td>
        <td><b>${row.f1_score_percent}%</b></td>
        <td>${row.accuracy_percent}%</td>
        <td>${row.psnr_db} dB</td>
      ` : ""}
    `;
    tableBody.appendChild(tr);

    // Thumbnails
    const thumbItem = document.createElement("div");
    thumbItem.className = "exp-thumb-item";
    thumbItem.innerHTML = `
      <img src="${row.thumbnail_url}" alt="BS ${row.block_size} C ${row.c}">
      <div class="exp-thumb-meta">BS ${row.block_size} | C ${row.c}</div>
    `;
    thumbItem.addEventListener("click", () => {
      openImageModal(row.thumbnail_url, `Kombinasi: Block Size ${row.block_size}, C = ${row.c}`);
    });
    thumbGrid.appendChild(thumbItem);
  });
}

// ==========================================
// TAB EVALUASI (EVALUATION & GROUND TRUTH)
// ==========================================
function initEvaluationTab() {
  const gtFileInput = document.getElementById("gt-file-input");
  const gtDropzone = document.getElementById("gt-dropzone");

  if (!gtDropzone || !gtFileInput) return;

  gtDropzone.addEventListener("click", () => gtFileInput.click());

  gtFileInput.addEventListener("change", async () => {
    if (gtFileInput.files.length > 0) {
      try {
        showToast("Mengunggah ground truth...", "info");
        await API.uploadGroundTruth(appState.currentImageId, gtFileInput.files[0]);
        showToast("Ground truth berhasil diunggah!", "success");
        triggerAutoProcess();
      } catch (err) {
        showToast(err.message, "error");
      }
    }
  });
}

function updateEvaluationView() {
  if (!appState.currentProcessResult) return;
  const res = appState.currentProcessResult;
  const evalContainer = document.getElementById("eval-metrics-container");
  const warningContainer = document.getElementById("eval-warning-box");
  const diffImgElem = document.getElementById("eval-diff-img");

  if (!res.has_ground_truth || !res.evaluation) {
    if (evalContainer) {
      evalContainer.innerHTML = `
        <div style="padding:1.5rem; text-align:center; color:var(--text-muted);">
          <p>Belum ada Ground Truth yang diunggah untuk citra ini.</p>
          <p style="font-size:0.8rem; margin-top:0.3rem;">Unggah berkas biner Ground Truth (PNG/JPG) di atas atau pilih sampel bawaan.</p>
        </div>
      `;
    }
    if (diffImgElem) diffImgElem.src = "";
    if (warningContainer) warningContainer.style.display = "none";
    return;
  }

  const ev = res.evaluation;
  const m = ev.metrics;
  const c = ev.counts;

  // Warning jika di-resize
  if (ev.was_resized && ev.warning) {
    warningContainer.style.display = "flex";
    warningContainer.textContent = `⚠️ ${ev.warning}`;
  } else {
    warningContainer.style.display = "none";
  }

  // Render Metrik
  evalContainer.innerHTML = `
    <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));">
      <div class="stat-box">
        <div class="stat-label">Precision</div>
        <div class="stat-value" style="color:var(--accent-teal);">${m.precision_percent}%</div>
        <small style="color:var(--text-muted); font-size:0.7rem;">TP / (TP + FP)</small>
      </div>
      <div class="stat-box">
        <div class="stat-label">Recall</div>
        <div class="stat-value" style="color:var(--color-info);">${m.recall_percent}%</div>
        <small style="color:var(--text-muted); font-size:0.7rem;">TP / (TP + FN)</small>
      </div>
      <div class="stat-box">
        <div class="stat-label">F1-Score</div>
        <div class="stat-value" style="color:var(--color-success); font-weight:800;">${m.f1_score_percent}%</div>
        <small style="color:var(--text-muted); font-size:0.7rem;">Harmonic Mean</small>
      </div>
      <div class="stat-box">
        <div class="stat-label">Accuracy</div>
        <div class="stat-value">${m.accuracy_percent}%</div>
        <small style="color:var(--text-muted); font-size:0.7rem;">(TP + TN) / Total</small>
      </div>
      <div class="stat-box">
        <div class="stat-label">PSNR</div>
        <div class="stat-value" style="color:var(--accent-amber);">${m.psnr_display}</div>
        <small style="color:var(--text-muted); font-size:0.7rem;">MSE: ${m.mse}</small>
      </div>
    </div>

    <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:0.75rem;">
      <b>Distribusi Piksel Matriks Kebingungan:</b>
      TP = ${c.tp.toLocaleString()} px | FP = ${c.fp.toLocaleString()} px | FN = ${c.fn.toLocaleString()} px | TN = ${c.tn.toLocaleString()} px
    </div>
  `;

  // Diff Map
  if (diffImgElem && res.images.diff_map) {
    diffImgElem.src = res.images.diff_map;
  }
}

// ==========================================
// MODAL PREVIEW
// ==========================================
function openImageModal(imgSrc, titleText) {
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");

  if (!modal || !modalImg) return;
  modalImg.src = imgSrc;
  if (modalTitle) modalTitle.textContent = titleText;
  modal.classList.add("open");
}

function initModal() {
  const modal = document.getElementById("image-modal");
  const closeBtn = document.getElementById("modal-close-btn");
  if (!modal) return;

  if (closeBtn) closeBtn.addEventListener("click", () => modal.classList.remove("open"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });
}

// ==========================================
// APP INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initAccordions();
  initViewButtons();
  initUpload();
  initExperiment();
  initEvaluationTab();
  initModal();

  // Inisialisasi Before/After Viewer
  appState.splitViewer = new SplitViewer("split-slider");

  // Bind controls
  bindSliderAndNumber("param-block-size", "input-block-size", "badge-block-size");
  bindSliderAndNumber("param-c", "input-c", "badge-c");
  bindSliderAndNumber("cmp-global-thresh", "input-cmp-global-thresh", "badge-cmp-global-thresh");

  document.getElementById("btn-run-cmp").addEventListener("click", runComparison);

  // Toggle listeners
  [
    "param-engine",
    "param-fg-color",
    "toggle-gaussian-blur",
    "param-gb-ksize",
    "param-gb-sigma",
    "toggle-median-blur",
    "param-mb-ksize",
    "toggle-clahe",
    "param-clahe-clip",
    "param-clahe-tile",
    "param-morph-op",
    "param-morph-ksize",
    "param-morph-iter",
    "param-morph-shape",
    "toggle-remove-small",
    "param-min-area",
    "param-remove-target"
  ].forEach((id) => {
    const elem = document.getElementById(id);
    if (elem) {
      elem.addEventListener("change", triggerAutoProcess);
      if (elem.type === "range") elem.addEventListener("input", triggerAutoProcess);
    }
  });

  // Load initial samples and trigger first process
  loadSamples().then(() => {
    triggerAutoProcess();
  });
});
