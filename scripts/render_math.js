/* Render a batch of TeX expressions to self-contained SVG without network I/O. */
const fs = require('node:fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const input = new TeX({
  packages: AllPackages,
  formatError: (_jax, error) => { throw error; },
});
const output = new SVG({fontCache: 'local'});
const document = mathjax.document('', {InputJax: input, OutputJax: output});

try {
  const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (!Array.isArray(formulas)) throw new Error('Expected a formula array');
  const result = formulas.map(({tex, display}) => {
    try {
      if (typeof tex !== 'string' || tex.length > 10000) {
        throw new Error('Invalid formula');
      }
      const node = document.convert(tex, {display: Boolean(display)});
      const svg = adaptor.outerHTML(adaptor.firstChild(node));
      // MathJax can encode TeX errors as SVG instead of throwing.
      if (!svg.startsWith('<svg') || svg.includes('data-mjx-error')) {
        throw new Error('TeX conversion failed');
      }
      return svg;
    } catch (error) {
      return null;
    }
  });
  process.stdout.write(JSON.stringify(result));
} catch (error) {
  process.stderr.write(`Math renderer failed: ${error.message}\n`);
  process.exitCode = 1;
}
