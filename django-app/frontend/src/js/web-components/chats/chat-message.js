// @ts-check

class ChatMessage extends HTMLElement {
  connectedCallback() {
    const uuid = crypto.randomUUID();
    this.innerHTML = `
            <div class="iai-chat-bubble iai-chat-bubble--${
              this.dataset.role === "user" ? "right" : "left"
            } govuk-body {{ classes }}" data-role="{{ role }}" tabindex="-1">
                <div class="iai-chat-bubble__header">
                    <div class="iai-chat-bubble__role">${
                      this.dataset.role === "ai" ? "Redbox" : "You"
                    }</div>
                </div>
                <markdown-converter class="iai-chat-bubble__text">${
                  this.dataset.text || ""
                }</markdown-converter>
                ${
                  !this.dataset.text
                    ? `
                      <div class="rb-loading-ellipsis govuk-body-s" aria-label="Loading message">
                        Loading
                        <span aria-hidden="true">.</span>
                        <span aria-hidden="true">.</span>
                        <span aria-hidden="true">.</span>
                      </div>
                      <div class="rb-loading-complete govuk-visually-hidden" aria-live="assertive"></div>
                      `
                    : ""
                }
                <sources-list></sources-list>
                <div class="govuk-notification-banner govuk-notification-banner--error govuk-!-margin-bottom-3 govuk-!-margin-top-3" role="alert" aria-labelledby="notification-title-${uuid}" data-module="govuk-notification-banner" hidden>
                    <div class="govuk-notification-banner__header">
                        <h3 class="govuk-notification-banner__title" id="notification-title-${uuid}">Error</h3>
                    </div>
                    <div class="govuk-notification-banner__content">
                        <p class="govuk-notification-banner__heading"></p>
                    </div>
                </div>
            </div>
            <feedback-buttons></feedback-buttons>
        `;

    // ensure new chat-messages aren't hidden behind the chat-input
    this.scrollIntoView({ block: "end" });

    // Insert route_display HTML
    const routeTemplate = /** @type {HTMLTemplateElement} */ (
      document.querySelector("#template-route-display")
    );
    const routeClone = document.importNode(routeTemplate.content, true);
    this.querySelector(".iai-chat-bubble__header")?.appendChild(routeClone);
  }

  /**
   * Streams an LLM response
   * @param {string} message
   * @param {string[]} selectedDocuments
   * @param {string | undefined} sessionId
   * @param {string} endPoint
   * @param {HTMLElement} chatControllerRef
   */
  stream = (
    message,
    selectedDocuments,
    sessionId,
    endPoint,
    chatControllerRef
  ) => {
    let responseContainer = /** @type MarkdownConverter */ (
      this.querySelector("markdown-converter")
    );
    let sourcesContainer = /** @type SourcesList */ (
      this.querySelector("sources-list")
    );
    let feedbackContainer = this.querySelector("feedback-buttons");
    let responseLoading = /** @type HTMLElement */ (
      this.querySelector(".rb-loading-ellipsis")
    );
    let responseComplete = this.querySelector(".rb-loading-complete");
    let webSocket = new WebSocket(endPoint);
    let streamedContent = "";
    let sources = [];

    // Stop streaming on escape key press
    this.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape" && this.dataset.status === "streaming") {
        this.dataset.status = "stopped";
        webSocket.close();
      }
    });

    webSocket.onopen = (event) => {
      webSocket.send(
        JSON.stringify({
          message: message,
          sessionId: sessionId,
          selectedFiles: selectedDocuments,
        })
      );
      this.dataset.status = "streaming";
    };

    webSocket.onerror = (event) => {
      responseContainer.innerHTML =
        "There was a problem. Please try sending this message again.";
      this.dataset.status = "error";
    };

    webSocket.onclose = (event) => {
      responseLoading.style.display = "none";
      if (responseComplete) {
        responseComplete.textContent = "Response complete";
      }
      if (this.dataset.status !== "stopped") {
        this.dataset.status = "complete";
      }
    };

    webSocket.onmessage = (event) => {
      let response;
      try {
        response = JSON.parse(event.data);
      } catch (err) {
        console.log("Error getting JSON response", err);
      }

      if (response.type === "text") {
        streamedContent += response.data;
        responseContainer.update(streamedContent);
      } else if (response.type === "session-id") {
        chatControllerRef.dataset.sessionId = response.data;
      } else if (response.type === "source") {
        sourcesContainer.add(
          response.data.original_file_name,
          response.data.url
        );
      } else if (response.type === "route") {
        let route = this.querySelector(".iai-chat-bubble__route");
        let routeText = route?.querySelector(".iai-chat-bubble__route-text");
        if (route && routeText) {
          routeText.textContent = response.data;
          route.removeAttribute("hidden");
        }

        // send route to Plausible
        let plausible = /** @type {any} */ (window).plausible;
        if (typeof plausible !== "undefined") {
          plausible("Chat-message-route", { props: { route: response.data } });
        }
      } else if (response.type === "hidden-route") {
        // TODO(@rachaelcodes): remove hidden-route with new route design
        // https://technologyprogramme.atlassian.net/browse/REDBOX-419

        // send route to Plausible
        let plausible = /** @type {any} */ (window).plausible;
        if (typeof plausible !== "undefined") {
          plausible("Chat-message-route", { props: { route: response.data } });
        }
      } else if (response.type === "end") {
        sourcesContainer.showCitations(response.data.message_id);
        feedbackContainer.showFeedback(response.data.message_id);
        const chatResponseEndEvent = new CustomEvent("chat-response-end", {
          detail: {
            title: response.data.title,
            session_id: response.data.session_id,
          },
        });
        document.dispatchEvent(chatResponseEndEvent);
      } else if (response.type === "error") {
        this.querySelector(".govuk-notification-banner")?.removeAttribute(
          "hidden"
        );
        this.querySelector(".govuk-notification-banner__heading").innerHTML =
          response.data;
      }

      // ensure new content isn't hidden behind the chat-input
      this.scrollIntoView({ block: "end" });
    };
  };
}
customElements.define("chat-message", ChatMessage);
