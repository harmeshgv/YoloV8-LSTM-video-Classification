import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/YoloV8-LSTM-video-Classification/', // <-- your repo name
  plugins: [react()],
})

