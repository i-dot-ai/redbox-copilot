// @ts-check

export class ChatTitle extends HTMLElement {
    connectedCallback() {
        this.innerHTML = `
      <div class="chat-title__heading-container">
        <div class="chat-title__heading-container-inner">
          ${
            this.dataset.title
                ? `
            <h2 class="chat-title__heading govuk-heading-m">${this.dataset.title}</h2>
          `
                : `
            <h2 class="chat-title__heading govuk-heading-s govuk-visually-hidden">Current chat</h2>
          `
        }
          <button class="chat-title__edit-btn" type="button">
            <svg width="16" height="16" viewBox="0 0 25 24" fill="none" aria-hidden="true" focusable="false">
              <path d="M11.9766 4H4.97656C4.44613 4 3.93742 4.21071 3.56235 4.58579C3.18728 4.96086 2.97656 5.46957 2.97656 6V20C2.97656 20.5304 3.18728 21.0391 3.56235 21.4142C3.93742 21.7893 4.44613 22 4.97656 22H18.9766C19.507 22 20.0157 21.7893 20.3908 21.4142C20.7658 21.0391 20.9766 20.5304 20.9766 20V13" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M19.4766 2.49998C19.8744 2.10216 20.414 1.87866 20.9766 1.87866C21.5392 1.87866 22.0787 2.10216 22.4766 2.49998C22.8744 2.89781 23.0979 3.43737 23.0979 3.99998C23.0979 4.56259 22.8744 5.10216 22.4766 5.49998L12.9766 15L8.97656 16L9.97656 12L19.4766 2.49998Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Edit
            <span class="govuk-visually-hidden"> chat title</span>
          </button>
        </div>
      </div>
      <div class="chat-title__form-container" hidden>
        <label for="chat-title" class="govuk-visually-hidden">Chat Title</label>
        <input type="text" class="chat-title__input" id="chat-title" maxlength="${
            this.dataset.titleLength
        }" value="${this.dataset.title}" tabindex="-1"/>
      </div>
    `;

        this.headingContainer = this.querySelector(
            ".chat-title__heading-container"
        );
        this.formContainer = this.querySelector(".chat-title__form-container");
        /** @type {HTMLButtonElement | null} */
        this.editButton = this.querySelector(".chat-title__edit-btn");
        /** @type {HTMLInputElement | null} */
        this.input = this.querySelector(".chat-title__input");
        this.heading = this.querySelector(".chat-title__heading");

        this.editButton?.addEventListener("click", this.showForm);
        this.heading?.addEventListener("click", this.showForm);
        this.input?.addEventListener("keydown", (e) => {
            if (!this.input) {
                return false;
            }
            switch (/** @type {KeyboardEvent} */ (e).key) {
                case "Escape":
                    this.input.value = this.dataset.title || "";
                    this.hideForm();
                    return true;
                case "Enter":
                    e.preventDefault();
                    this.update();
                    return true;
                default:
                    return true;
            }
        });
        this.input?.addEventListener("change", (e) => {
            this.update();
        });
        this.input?.addEventListener("blur", (e) => {
            this.update();
        });

        if (!this.dataset.sessionId) {
            document.addEventListener("chat-response-end", this.onFirstResponse);
        }
    }

    showForm = () => {
        this.headingContainer?.setAttribute("hidden", "");
        this.formContainer?.removeAttribute("hidden");
        this.input?.focus();
    };

    hideForm = () => {
        this.headingContainer?.removeAttribute("hidden");
        this.formContainer?.setAttribute("hidden", "");
        this.editButton?.focus();
    };

    onFirstResponse = (e) => {
        this.dataset.title = e.detail.title;
        this.dataset.sessionId = e.detail.session_id;
        document.removeEventListener("chat-response-end", this.onFirstResponse);
        if (this.input && this.heading) {
            this.input.value = e.detail.title;
            this.heading.textContent = `${e.detail.title}`;
            this.heading.classList.remove("govuk-visually-hidden");
        }
    };

    update = () => {
        const newTitle = this.input?.value;
        this.send(newTitle);
        this.dataset.title = newTitle;
        if (this.heading) {
            this.heading.textContent = newTitle || "";
        }
        this.hideForm();
    };

    send = (newTitle) => {
        const csrfToken =
            /** @type {HTMLInputElement | null} */ (
            document.querySelector('[name="csrfmiddlewaretoken"]')
        )?.value || "";
        fetch(`/chat/${this.dataset.sessionId}/title/`, {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-CSRFToken": csrfToken},
            body: JSON.stringify({name: newTitle}),
        });
    };
}

customElements.define("chat-title", ChatTitle);
