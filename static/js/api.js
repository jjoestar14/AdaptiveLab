/**
 * AdaptiveLab - API Client Module (static/js/api.js)
 * Menyediakan fungsi-fungsi asynchronous untuk berkomunikasi dengan backend Flask.
 */

const API = {
  /**
   * Mengambil daftar sampel bawaan.
   */
  async getSamples() {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal mengambil daftar sampel.");
    }
    return data.samples;
  },

  /**
   * Mengunggah citra baru.
   */
  async uploadImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal mengunggah citra.");
    }
    return data;
  },

  /**
   * Mengunggah citra Ground Truth untuk image_id tertentu.
   */
  async uploadGroundTruth(imageId, file) {
    const formData = new FormData();
    formData.append("image_id", imageId);
    formData.append("file", file);

    const res = await fetch("/api/upload-gt", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal mengunggah ground truth.");
    }
    return data;
  },

  /**
   * Memproses citra tunggal dengan parameter lengkap.
   */
  async processImage(payload) {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal memproses citra.");
    }
    return data;
  },

  /**
   * Membandingkan 4 metode segmentasi biner.
   */
  async compareMethods(payload) {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal membandingkan metode.");
    }
    return data.methods;
  },

  /**
   * Menjalankan eksperimen grid search.
   */
  async runExperiment(payload) {
    const res = await fetch("/api/experiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Gagal menjalankan eksperimen.");
    }
    return data;
  }
};
