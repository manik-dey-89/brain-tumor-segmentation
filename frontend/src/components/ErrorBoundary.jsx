import React from 'react'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'
import { Link } from 'react-router-dom'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary] Caught render error:', error, errorInfo)
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null })
    if (typeof window !== 'undefined') window.location.reload()
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    if (this.props.onReset) { try { this.props.onReset() } catch {} }
  }

  render() {
    if (this.state.hasError) {
      const msg = this.state.error?.message || 'An unexpected rendering error occurred'
      return (
        <div className="min-h-[60vh] flex items-center justify-center px-4">
          <div className="max-w-xl w-full glass-card p-6 sm:p-8 animate-fade-in">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-2xl bg-red-50 border border-red-200 flex-shrink-0">
                <AlertTriangle className="w-7 h-7 text-red-500" />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-xl font-bold text-red-700">
                  {this.props.title || 'Rendering Error'}
                </h2>
                <p className="text-sm text-slate-500 mt-2 leading-relaxed">
                  {this.props.message || 'The application encountered an error while rendering this section.'}
                </p>
                <div className="mt-4 p-3 rounded-xl bg-slate-50 border border-slate-200
                                font-mono text-xs text-red-600 break-words max-h-36 overflow-y-auto">
                  {msg}
                  {this.state.error?.stack && (
                    <details className="mt-2 text-slate-400">
                      <summary className="cursor-pointer hover:text-slate-600">Stack trace</summary>
                      <pre className="mt-1 whitespace-pre-wrap text-[10px]">
                        {this.state.error.stack}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-6 flex flex-wrap gap-2">
              <button onClick={this.handleReload} className="btn-primary flex items-center gap-2 text-sm">
                <RefreshCw className="w-4 h-4" />
                Reload Page
              </button>
              {this.props.onReset && (
                <button onClick={this.handleReset} className="btn-secondary flex items-center gap-2 text-sm">
                  Reset State
                </button>
              )}
              <Link to="/" className="btn-secondary flex items-center gap-2 text-sm">
                <Home className="w-4 h-4" />
                Back to Home
              </Link>
            </div>
            <p className="mt-5 text-[11px] text-slate-400 text-center">
              {this.props.footer || 'If this error persists, please check the browser console for details.'}
            </p>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
