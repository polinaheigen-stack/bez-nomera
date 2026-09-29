import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react()],
  resolve: {
    dedupe: ['react', 'react-dom', 'lucide-react'],
  },
  server: {
    host: '127.0.0.1', port: 3017, strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8017' },
    fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] },
  },
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: false },
});
