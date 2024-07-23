// @ts-check

class CopyText extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
        <button class="iai-chat-bubble__citations-button" type="button">
          <svg width="19" height="18" viewBox="0 0 19 18" fill="none" focusable="false" aria-hidden="true">
            <path d="M6.875 3H5C4.17157 3 3.5 3.67157 3.5 4.5V15C3.5 15.8284 4.17157 16.5 5 16.5H14C14.8284 16.5 15.5 15.8284 15.5 15V4.5C15.5 3.67157 14.8284 3 14 3H12.125" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M6.5 4.8V3.375C6.5 3.16789 6.66789 3 6.875 3C7.08211 3 7.25317 2.83203 7.28864 2.62798C7.39976 1.98878 7.83049 0.75 9.5 0.75C11.1695 0.75 11.6002 1.98878 11.7114 2.62798C11.7468 2.83203 11.9179 3 12.125 3C12.3321 3 12.5 3.16789 12.5 3.375V4.8C12.5 5.04853 12.2985 5.25 12.05 5.25H6.95C6.70147 5.25 6.5 5.04853 6.5 4.8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          Copy to clipboard
        </button>
    `;

    const copyToClip = (str) => {
      function listener(evt) {
        evt.clipboardData.setData("text/html", str);
        evt.clipboardData.setData("text/plain", str);
        evt.preventDefault();
      }
      document.addEventListener("copy", listener);
      document.execCommand("copy");
      document.removeEventListener("copy", listener);
    };

    this.querySelector("button")?.addEventListener("click", () => {
      const textEl = this.closest(".iai-chat-bubble")?.querySelector(
        ".iai-chat-bubble__text"
      );
      copyToClip(textEl?.innerHTML);
    });
  }
}
customElements.define("copy-text", CopyText);
