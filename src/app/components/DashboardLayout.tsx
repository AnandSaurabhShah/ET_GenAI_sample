import React from 'react';
import { Outlet, Link, useLocation } from 'react-router';
import { liveTools, previewFeatures } from '../data/features';
import { cn } from './ui';

export function DashboardLayout() {
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-[#F9F8F4] font-sans text-slate-900">
      {/* Sidebar */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col h-screen sticky top-0 overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <Link to="/dashboard" className="flex items-center gap-2 group">
             <div className="w-8 h-8 bg-[#115E59] rounded-md flex items-center justify-center">
               <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
             </div>
             <h1 className="font-serif font-bold text-xl text-slate-900 group-hover:text-[#115E59] transition-colors">Enterprise AI</h1>
          </Link>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-8">
          
          <nav>
             <Link 
                to="/dashboard" 
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  location.pathname === '/dashboard' 
                    ? "bg-[#115E59]/10 text-[#115E59]" 
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                )}
             >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                Dashboard Home
             </Link>
          </nav>

          <div>
            <h2 className="px-3 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Live Tools</h2>
            <div className="space-y-0.5">
              {liveTools.map(t => {
                const isActive = location.pathname === `/dashboard/${t.id}`;
                return (
                  <Link 
                    key={t.id} 
                    to={`/dashboard/${t.id}`} 
                    className={cn(
                      "block px-3 py-2 rounded-md text-sm transition-colors",
                      isActive 
                        ? "bg-[#115E59] text-white font-medium shadow-sm" 
                        : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                    )}
                  >
                    {t.title}
                  </Link>
                );
              })}
            </div>
          </div>

          <div>
            <h2 className="px-3 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Platform Preview</h2>
            <div className="space-y-0.5">
              {previewFeatures.map(p => {
                const isActive = location.pathname === `/dashboard/${p.id}`;
                return (
                  <Link 
                    key={p.id} 
                    to={`/dashboard/${p.id}`} 
                    className={cn(
                      "flex justify-between items-center px-3 py-2 rounded-md text-sm transition-colors",
                      isActive 
                        ? "bg-orange-100 text-orange-900 font-medium" 
                        : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
                    )}
                  >
                    <span className="truncate">{p.title}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        </div>

        {/* Made in India Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50">
          <div className="flex flex-col items-center justify-center text-center space-y-1">
            <span className="text-xl">🇮🇳</span>
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Made in India</span>
            <span className="text-xs font-medium text-slate-700">Built for the World</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto h-screen">
        <Outlet />
      </main>
    </div>
  );
}
