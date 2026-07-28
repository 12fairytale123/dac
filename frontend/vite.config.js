import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发服务器默认 5173；/api 代理到后端 8000，避免跨域配置
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
