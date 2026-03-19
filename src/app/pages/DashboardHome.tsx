import React from 'react';
import { Card, Badge, Button } from '../components/ui';
import { Link } from 'react-router';
import { liveTools, previewFeatures } from '../data/features';

export function DashboardHome() {
  return (
    <div className="pb-20 bg-[#F9F8F4]">
      {/* Hero Section */}
      <section className="mx-auto max-w-7xl px-8 pt-16">
        <div className="grid gap-12 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-orange-100 border border-orange-200 rounded-full text-xs font-bold tracking-wider text-orange-800 mb-6">
              <span>🏆</span>
              ECONOMIC TIMES GENAI HACKATHON
            </div>
            <h1 className="font-serif text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl text-slate-900">
              Enterprise GenAI: Architected in India. Scaling the Global Fortune 500.
            </h1>
            <p className="mt-6 text-lg text-slate-600 max-w-2xl leading-relaxed">
              A highly-scalable, multi-agent platform designed to solve complex business operations. Combining Indic-first models with global DPA/GDPR compliance for massive ROI.
            </p>
          </div>
          
          <div className="grid grid-cols-2 gap-4 lg:col-span-1">
            <Card className="flex flex-col items-center justify-center p-6 text-center shadow-sm hover:shadow-md transition-shadow">
              <span className="text-xs font-bold text-slate-500 tracking-wider uppercase">Architecture</span>
              <span className="mt-2 font-medium">Billion-Scale Ready</span>
            </Card>
            <Card className="flex flex-col items-center justify-center p-6 text-center shadow-sm hover:shadow-md transition-shadow">
              <span className="text-xs font-bold text-slate-500 tracking-wider uppercase">Compliance</span>
              <span className="mt-2 font-medium">DPDP / GDPR</span>
            </Card>
            <Card className="flex flex-col items-center justify-center p-6 text-center shadow-sm hover:shadow-md transition-shadow border-[#115E59]/20 bg-[#115E59]/5">
              <span className="text-xs font-bold text-[#115E59] tracking-wider uppercase">Live ROI Tools</span>
              <span className="mt-2 font-medium text-slate-900">9 Active</span>
            </Card>
            <Card className="flex flex-col items-center justify-center p-6 text-center shadow-sm hover:shadow-md transition-shadow">
              <span className="text-xs font-bold text-slate-500 tracking-wider uppercase">Foundation</span>
              <span className="mt-2 font-medium">Indic + Global Models</span>
            </Card>
          </div>
        </div>
      </section>

      {/* Telemetry Bar */}
      <section className="mx-auto max-w-7xl px-8 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-t border-slate-200 py-3 text-xs font-bold tracking-wider text-slate-500 bg-white/50 backdrop-blur-sm px-4 rounded-lg">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-slate-700">CLUSTER: AP-SOUTH-1 (MUMBAI)</span>
            </div>
            <span className="text-slate-300">|</span>
            <div className="text-slate-700">THROUGHPUT: 12k REQ/S</div>
            <span className="text-slate-300">|</span>
            <div className="text-slate-700">LATENCY: 42ms</div>
          </div>
          <div className="flex items-center gap-2 bg-[#115E59]/10 text-[#115E59] px-3 py-1 rounded-full">
            <span>Enterprise PPO Target Engaged</span>
          </div>
        </div>
      </section>

      {/* Interactive Live Tools */}
      <section className="mx-auto max-w-7xl px-8 py-8">
        <div className="mb-8 flex justify-between items-end">
          <div>
            <h2 className="font-serif text-3xl font-bold text-slate-900">Live Interactive Tools</h2>
            <p className="mt-2 text-slate-600">Test core platform capabilities directly in the browser.</p>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {liveTools.map((tool) => (
            <Card key={tool.id} className="flex flex-col p-6 hover:border-[#115E59]/50 transition-colors group relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent to-slate-50/50 opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="mb-4 flex items-start justify-between relative z-10">
                <div>
                  <span className="text-xs font-bold text-[#115E59] uppercase tracking-wider">{tool.category}</span>
                  <h3 className="mt-1 text-xl font-bold text-slate-900">{tool.title}</h3>
                </div>
                <Badge variant="live">LIVE</Badge>
              </div>
              <p className="text-sm text-slate-600 mb-6 flex-1 relative z-10">{tool.description}</p>
              <div className="relative z-10">
                <Link to={`/dashboard/${tool.id}`}>
                   <Button className="w-full justify-between group-hover:bg-[#0f4d49]">
                      Launch Tool
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                   </Button>
                </Link>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* Product Suite Overview */}
      <section className="mx-auto max-w-7xl px-8 py-16">
        <div className="mb-10">
          <h2 className="font-serif text-3xl font-bold text-slate-900">Twenty modules organized like a product suite</h2>
          <p className="mt-2 text-slate-600">Additional platform capabilities in active development or lazy-loaded.</p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {previewFeatures.map((feature, idx) => (
            <Card key={idx} className="flex flex-col p-5 h-full relative group hover:border-orange-200 transition-colors bg-white">
              <div className="absolute top-4 right-4">
                <Badge variant="preview">PREVIEW</Badge>
              </div>
              <h3 className="mt-2 text-lg font-bold pr-16 text-slate-800 group-hover:text-slate-900">{feature.title}</h3>
              <p className="mt-3 text-sm text-slate-500 flex-1">{feature.description}</p>
              <Link to={`/dashboard/${feature.id}`} className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs font-mono text-slate-400 font-semibold uppercase group-hover:text-orange-600 transition-colors">
                <span>Status: Lazy</span>
                <span className="opacity-0 group-hover:opacity-100 transition-opacity">View Details →</span>
              </Link>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
