// @ts-check

class MessageInput extends HTMLElement {

    connectedCallback() {

        const textarea = this.querySelector('textarea');
        if (!textarea) {
            return;
        }

        // Submit form on enter-key press (providing shift isn't being pressed)
        textarea.addEventListener('keypress', (evt) => {
            if (evt.key === 'Enter' && !evt.shiftKey) {
                evt.preventDefault();
                if (textarea.value.trim()) {
                    this.closest('form')?.requestSubmit();
                }
            }
        });

        // expand textarea as user adds lines
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight}px`;
        });

    }
  
}
customElements.define('message-input', MessageInput);