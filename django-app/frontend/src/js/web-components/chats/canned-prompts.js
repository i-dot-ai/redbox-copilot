// @ts-check

class CannedPrompts extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
        <h3 class="chat-options__heading govuk-heading-m">What would you like to ask your Redbox?</h3>
        <div class="chat-options__options">
            <button class="chat-options__option chat-options__option_agenda" type="button">
                <img src="/static/icons/icon_square_doc.svg" alt=""/>
                Draft an agenda for a team meeting
            </button>
            <button class="chat-options__option chat-options__option_objectives" type="button">
                <img src="/static/icons/archery.svg" alt=""/>
                Help me set my work objectives
            </button>
            <button class="chat-options__option chat-options__option_ps_role" type="button">
                <img src="/static/icons/person.svg" alt=""/>
                Describe the role of a Permanent Secretary
            </button>
        </div>
        <p class="chat-options__info-text">Or type any question below</p>
        `;

    this.querySelector(".chat-options__option_agenda")?.addEventListener(
      "click",
      (e) => {
        this.prepopulateMessageBox("Draft an agenda for a team meeting");
      }
    );
    this.querySelector(".chat-options__option_objectives")?.addEventListener(
      "click",
      (e) => {
        this.prepopulateMessageBox("Help me set my work objectives");
      }
    );
    this.querySelector(".chat-options__option_ps_role")?.addEventListener(
      "click",
      (e) => {
        this.prepopulateMessageBox(
          "Describe the role of a Permanent Secretary"
        );
      }
    );

    window.setTimeout(() => {
      this.scrollIntoView({ block: "end" });
    }, 100);
  }

  prepopulateMessageBox = (prompt) => {
    /** @type HTMLInputElement | null */
    let chatInput = document.querySelector(".iai-chat-input__input");
    if (chatInput) {
      chatInput.value = prompt;
      chatInput.focus();
      chatInput.selectionStart = chatInput.value.length;
    }
  };
}
customElements.define("canned-prompts", CannedPrompts);
