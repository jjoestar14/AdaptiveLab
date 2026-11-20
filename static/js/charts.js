/**
 * AdaptiveLab - Histogram Chart Module (static/js/charts.js)
 * Menggunakan pustaka Chart.js untuk menampilkan kurva intensitas grayscale 256-bin.
 */

let histogramChartInstance = null;

function renderHistogram(canvasId, histData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return;

  const isDarkMode = document.documentElement.getAttribute("data-theme") === "dark" ||
    (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);

  const gridColor = isDarkMode ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  const textColor = isDarkMode ? "#94a3b8" : "#64748b";
  const barColor = isDarkMode ? "rgba(20, 184, 166, 0.7)" : "rgba(13, 148, 136, 0.7)";

  if (histogramChartInstance) {
    histogramChartInstance.destroy();
  }

  const ctx = canvas.getContext("2d");
  histogramChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: histData.labels,
      datasets: [{
        label: "Frekuensi Piksel",
        data: histData.counts,
        backgroundColor: barColor,
        borderColor: "transparent",
        borderWidth: 0,
        barPercentage: 1.0,
        categoryPercentage: 1.0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Intensitas: ${items[0].label}`,
            label: (item) => `Jumlah: ${item.raw.toLocaleString()} piksel`
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { display: false },
          ticks: {
            color: textColor,
            maxTicksLimit: 9,
            font: { size: 10 }
          },
          title: {
            display: true,
            text: "Tingkat Keabuan (0 = Gelap, 255 = Terang)",
            color: textColor,
            font: { size: 10 }
          }
        },
        y: {
          display: true,
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            maxTicksLimit: 5,
            font: { size: 10 }
          }
        }
      }
    }
  });
}
