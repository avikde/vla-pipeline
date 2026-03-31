#!/usr/bin/env node
/** Local dev server with COOP/COEP headers for MuJoCo WASM (SharedArrayBuffer). */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Auto-setup: download mujoco.js + mujoco.wasm from npm if missing
if (!fs.existsSync(path.join(__dirname, 'mujoco.wasm'))) {
  console.log('mujoco.wasm not found — running npm run setup...');
  execSync('npm run setup', { cwd: __dirname, stdio: 'inherit' });
}

const PORT = process.argv[2] || 8080;
const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.wasm': 'application/wasm',
  '.xml': 'application/xml', '.stl': 'application/octet-stream', '.png': 'image/png',
  '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
};

const server = http.createServer((req, res) => {
  let filePath = path.join(__dirname, req.url === '/' ? '/index.html' : req.url);
  filePath = decodeURIComponent(filePath.split('?')[0]);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Serving web/ on http://localhost:${PORT} (with COOP/COEP headers)`);
});
