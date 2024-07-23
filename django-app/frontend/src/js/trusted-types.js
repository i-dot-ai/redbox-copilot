// @ts-check

/** @type {import ('../../node_modules/@types/dompurify/index.d.ts')} */
let DOMPurify = window["DOMPurify"];

// Create default policy
if (typeof window.trustedTypes !== "undefined") {
  window.trustedTypes.createPolicy("default", {
    createHTML: (to_escape) =>
      DOMPurify.sanitize(to_escape, {
        RETURN_TRUSTED_TYPE: false,
        CUSTOM_ELEMENT_HANDLING: {
          tagNameCheck: (tagName) =>
            tagName === "markdown-converter" ||
            tagName === "sources-list" ||
            tagName === "tool-tip" ||
            tagName === "feedback-buttons" ||
            tagName === "chat-title" ||
            tagName === "copy-text",
          attributeNameCheck: (attr) => true,
          allowCustomizedBuiltInElements: true,
        },
      }),
  });
}
