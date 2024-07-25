// @ts-check

class FileStatus extends HTMLElement {
  connectedCallback() {
    const checkStatus = async () => {
      // UPDATE THESE AS REQUIRED
      const FILE_STATUS_ENDPOINT = "/file-status";
      const CHECK_INTERVAL_MS = 5000;

      // as we use different layouts for mobile and desktop - only request update if visible
      if (!this.checkVisibility()) {
        window.setTimeout(checkStatus, CHECK_INTERVAL_MS);
        return;
      }

      const response = await fetch(
        `${FILE_STATUS_ENDPOINT}?id=${this.dataset.id}`
      );
      const responseObj = await response.json();
      this.textContent = responseObj.status;
      if (responseObj.status.toLowerCase() === "complete") {
        const evt = new CustomEvent("doc-complete", {
          detail: this.closest(".iai-doc-list__item"),
        });
        document.body.dispatchEvent(evt);
      } else {
        window.setTimeout(checkStatus, CHECK_INTERVAL_MS);
      }
    };

    checkStatus();
  }
}
customElements.define("file-status", FileStatus);
