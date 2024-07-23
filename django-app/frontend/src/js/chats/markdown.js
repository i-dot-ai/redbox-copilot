// @ts-check

/** @type {import ('../../node_modules/@types/showdown/index.d.ts')} */
let showdown = window["showdown"];


class MarkdownConverter extends HTMLElement {
    
    /**
     * Takes markdown and inserts as HTML
     * @param {string} markdown 
     */
    update(markdown) {
        let converter = new showdown.Converter({
            disableForced4SpacesIndentedSublists: true,
            headerLevelStart: 3,
            tables: true
        });
        this.innerHTML = converter.makeHtml(markdown);
    }

    connectedCallback() {
        this.update(this.textContent || '');
    }

    
}
customElements.define('markdown-converter', MarkdownConverter);
