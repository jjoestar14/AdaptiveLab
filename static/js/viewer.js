/**
 * AdaptiveLab - Image Viewer & Before/After Comparison Slider (static/js/viewer.js)
 */

class SplitViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.beforeImg = this.container.querySelector(".split-image-before");
    this.afterImg = this.container.querySelector(".split-image-after");
    this.divider = this.container.querySelector(".split-divider");
    this.isDragging = false;

    this.initEvents();
  }

  initEvents() {
    const startDrag = (e) => {
      this.isDragging = true;
      this.onDrag(e);
    };

    const stopDrag = () => {
      this.isDragging = false;
    };

    this.container.addEventListener("mousedown", startDrag);
    window.addEventListener("mouseup", stopDrag);
    window.addEventListener("mousemove", (e) => {
      if (this.isDragging) this.onDrag(e);
    });

    // Touch events for mobile
    this.container.addEventListener("touchstart", startDrag, { passive: true });
    window.addEventListener("touchend", stopDrag);
    window.addEventListener("touchmove", (e) => {
      if (this.isDragging && e.touches.length > 0) {
        this.onDrag(e.touches[0]);
      }
    });
  }

  onDrag(e) {
    const rect = this.container.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    this.divider.style.left = `${percent}%`;
    this.afterImg.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
  }

  setImages(beforeSrc, afterSrc) {
    if (this.beforeImg) this.beforeImg.src = beforeSrc;
    if (this.afterImg) this.afterImg.src = afterSrc;
  }

  resetPosition() {
    if (this.divider && this.afterImg) {
      this.divider.style.left = "50%";
      this.afterImg.style.clipPath = "polygon(50% 0, 100% 0, 100% 100%, 50% 100%)";
    }
  }
}
