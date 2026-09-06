import React, { useState } from 'react'
import { Outlet, NavLink, Link } from 'react-router-dom'
import { Brain, Clock, Menu, X, Activity } from 'lucide-react'
import ErrorBoundary from './ErrorBoundary'

const navItems = [
  { to: '/',        label: 'Home',    icon: Brain    },
  { to: '/analyze', label: 'Analyze', icon: Activity },
  { to: '/history', label: 'History', icon: Clock    },
]

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">

      {/* ── Navbar ─────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur-xl shadow-sm">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-medical-600/10 border border-medical-500/30
                            flex items-center justify-center
                            group-hover:bg-medical-600/20 transition-colors">
              <Brain className="w-5 h-5 text-medical-600" />
            </div>
            <span className="font-bold text-lg tracking-tight">
              <span className="text-medical-600">Brain Tumor</span>
              <span className="text-slate-800"> Segmentation</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                   transition-all duration-200 ${
                    isActive
                      ? 'bg-medical-50 text-medical-600 border border-medical-200'
                      : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </div>

          {/* Disclaimer badge */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full
                          bg-yellow-50 border border-yellow-200 text-yellow-700
                          text-xs font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
            Research Tool — Not Medical Advice
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-xl hover:bg-slate-100 transition-colors text-slate-600"
            onClick={() => setMobileOpen((v) => !v)}
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </nav>

        {/* Mobile menu */}
        {mobileOpen && (
          <div className="md:hidden border-t border-slate-200 px-4 py-3 space-y-1 bg-white">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium
                   transition-colors ${
                    isActive
                      ? 'bg-medical-50 text-medical-600'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </div>
        )}
      </header>

      {/* ── Page content ──────────────────────────────────────────── */}
      <main className="flex-1">
        <ErrorBoundary
          title="Application Error"
          message="Brain Tumor Segmentation encountered an unexpected error. Your current page could not be rendered. Please try reloading or returning home."
        >
          <Outlet />
        </ErrorBoundary>
      </main>

      {/* ── Footer ────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 bg-white py-6 px-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center
                        justify-between gap-4 text-sm text-slate-400">
          <span className="font-medium text-slate-500">
             Brain Tumor Segmentation Using U-Net – AI Research System
          </span>
          <span className="text-center text-slate-400">
            Research and educational use only. Not a medical diagnosis device.
          </span>
        </div>
      </footer>
    </div>
  )
}
