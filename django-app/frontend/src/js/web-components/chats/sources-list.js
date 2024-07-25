// @ts-check

class SourcesList extends HTMLElement {
  constructor() {
    super();
    this.sources = [];
  }

  /**
   * Adds a source to the current message
   * @param {string} fileName
   * @param {string} url
   */
  add = (fileName, url) => {
    this.sources.push({
      fileName: fileName,
      url: url,
    });

    let html = `
            <h3 class="iai-chat-bubble__sources-heading govuk-heading-s govuk-!-margin-bottom-1">Sources</h3>
            <div class="iai-display-flex-from-desktop">
            <ul class="govuk-list govuk-list--bullet govuk-!-margin-bottom-0">
        `;
    this.sources.forEach((source) => {
      html += `
                <li class="govuk-!-margin-bottom-0">
                    <a class="iai-chat-bubbles__sources-link govuk-link" href="${source.url}">${source.fileName}</a>
                </li>
            `;
    });
    html += `</div></ul>`;

    this.innerHTML = html;
  };

  /**
   * Shows to citations link/button
   * @param {string} chatId
   */
  showCitations = (chatId) => {
    if (!chatId) {
      return;
    }
    if (this.sources.length) {
      const html = `
        <div class="iai-chat-bubble__citations-button-container">
          <copy-text></copy-text>
          <a class="iai-chat-bubble__citations-button" href="/citations/${chatId}">
            <svg width="20" height="19" viewBox="0 0 20 19" fill="none" focusable="false" aria-hidden="true">
                <path d="M1.5 9.62502C1.5 9.62502 4.59036 3.55359 10 3.55359C15.4084 3.55359 18.5 9.62502 18.5 9.62502C18.5 9.62502 15.4084 15.6964 10 15.6964C4.59036 15.6964 1.5 9.62502 1.5 9.62502Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9.99993 10.8392C10.322 10.8392 10.6308 10.7113 10.8586 10.4836C11.0863 10.2558 11.2142 9.94698 11.2142 9.62493C11.2142 9.30288 11.0863 8.99402 10.8586 8.7663C10.6308 8.53858 10.322 8.41064 9.99993 8.41064C9.67788 8.41064 9.36902 8.53858 9.1413 8.7663C8.91358 8.99402 8.78564 9.30288 8.78564 9.62493C8.78564 9.94698 8.91358 10.2558 9.1413 10.4836C9.36902 10.7113 9.67788 10.8392 9.99993 10.8392Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            See the response info
          </a>
        </div>
      `;
      /** @type {HTMLElement} */ (
        this.querySelector(".iai-display-flex-from-desktop")
      ).innerHTML += html;
    } else {
      this.innerHTML = `<copy-text></copy-text>`;
    }
  };
}
customElements.define("sources-list", SourcesList);
