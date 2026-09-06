/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        medical: {
          50:  '#eef6ff',
          100: '#d9ecff',
          200: '#bbdeff',
          300: '#8acbff',
          400: '#50adff',
          500: '#2d8eff',
          600: '#156ef5',
          700: '#0d58e1',
          800: '#1147b6',
          900: '#13408f',
          950: '#0e2857',
        },
        // Light-theme neutrals
        neural: {
          50:  '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#f8fafc',   // kept as alias so bg-neural-950 = page background
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'scan':       'scan 2s ease-in-out infinite',
        'fade-in':    'fadeIn 0.4s ease-out',
        'slide-up':   'slideUp 0.4s ease-out',
      },
      keyframes: {
        scan: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':      { transform: 'translateY(100%)' },
        },
        fadeIn:  { from: { opacity: 0 },                                     to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(16px)' },
                   to:   { opacity: 1, transform: 'translateY(0)' } },
      },
      backdropBlur: { xs: '2px' },
    },
  },
  plugins: [],
}
