// @ts-check

/** So completed docs can be added to this list */
class DocList extends HTMLElement {
  connectedCallback() {
    document.body.addEventListener("doc-complete", (evt) => {
      const completedDoc = /** @type{CustomEvent} */ (evt).detail;
      this.querySelector("tbody")?.appendChild(completedDoc);
    });
  }
}
customElements.define("doc-list", DocList);
