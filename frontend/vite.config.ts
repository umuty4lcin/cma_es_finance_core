import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite dev sunucusu (5173) -> FastAPI backend (8000) proxy
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
