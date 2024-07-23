// IgnoreAssetResolver.js
const { Resolver } = require("@parcel/plugin");

module.exports = new Resolver({
  async resolve({ dependency }) {
    // If the dependency is an asset (like image, font, etc.), ignore it
    if (
      dependency.specifier.endsWith(".jpg") ||
      dependency.specifier.endsWith(".png") ||
      dependency.specifier.endsWith(".gif") ||
      dependency.specifier.endsWith(".svg") ||
      dependency.specifier.endsWith(".woff") ||
      dependency.specifier.endsWith(".woff2") ||
      dependency.specifier.endsWith(".eot") ||
      dependency.specifier.endsWith(".ttf") ||
      dependency.specifier.endsWith(".otf")
    ) {
      return {
        isExcluded: true,
      };
    }

    return null;
  },
});
