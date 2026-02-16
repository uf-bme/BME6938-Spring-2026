import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// import katex from 'katex';
import 'katex/dist/katex.min.css'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
