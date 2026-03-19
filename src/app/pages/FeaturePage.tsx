import React from 'react';
import { useParams, Link } from 'react-router';
import { liveTools, previewFeatures } from '../data/features';
import { Card, Badge, Button, Input, Textarea, Select, Terminal, cn } from '../components/ui';

export function FeaturePage() {
  const { featureId } = useParams();

  const tool = liveTools.find(t => t.id === featureId);
  const preview = previewFeatures.find(p => p.id === featureId);

  const renderToolContent = (id: string) => {
    switch (id) {
      case "code-mixed":
        return (
          <div className="space-y-4">
            <Input placeholder="Enter Hinglish, Tanglish, or global text..." />
            <Button className="w-full">Analyze global intent</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "sentiment":
        return (
          <div className="space-y-4">
            <Input placeholder="Enter customer feedback..." />
            <div className="px-1">
              <input type="range" className="w-full accent-[#115E59]" />
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>Negative</span>
                <span>Positive</span>
              </div>
            </div>
            <Button className="w-full">Score sentiment</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "meeting":
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-8 text-center hover:bg-slate-100 transition-colors">
              <input type="file" id="file1" className="hidden" />
              <label htmlFor="file1" className="cursor-pointer flex flex-col items-center justify-center gap-2">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                <span className="text-sm font-medium text-[#115E59] hover:underline">Choose audio/transcript file</span>
                <span className="text-xs text-slate-500">MP3, WAV, or TXT up to 50MB</span>
              </label>
            </div>
            <Button className="w-full">Analyze meeting</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "invoice":
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-8 text-center hover:bg-slate-100 transition-colors">
              <input type="file" id="file2" className="hidden" />
              <label htmlFor="file2" className="cursor-pointer flex flex-col items-center justify-center gap-2">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                <span className="text-sm font-medium text-[#115E59] hover:underline">Choose PDF or Image</span>
                <span className="text-xs text-slate-500">PDF, PNG, JPG up to 10MB</span>
              </label>
            </div>
            <Button className="w-full">Parse invoice</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "kyc":
        return (
          <div className="space-y-4">
            <Textarea placeholder="Paste raw Aadhaar, PAN, or Global Corporate ID text..." className="min-h-[160px]" />
            <Button className="w-full">Extract global entities</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "sla":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Select defaultValue="" options={["Low", "Medium", "High", "Critical"]} placeholder="Priority" />
              <Select defaultValue="" options={["Support", "Billing", "Technical", "Sales"]} placeholder="Category" />
              <Select defaultValue="" options={["Standard", "Premium", "Enterprise"]} placeholder="Customer tier" />
              <Select defaultValue="" options={["Simple", "Moderate", "Complex"]} placeholder="Complexity" />
            </div>
            <Button className="w-full">Predict risk</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "rag":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Input placeholder="Document title" />
              <Input placeholder="Category" />
            </div>
            <Textarea placeholder="Paste document content to embed..." className="min-h-[120px]" />
            <Button variant="outline" className="w-full">Add document to vector store</Button>
            
            <div className="h-px bg-slate-200 w-full my-6"></div>
            
            <div className="flex gap-2">
              <Input placeholder="Search query..." />
              <Button className="shrink-0">Search Memory</Button>
            </div>
            <Terminal className="mt-6" />
          </div>
        );
      case "gstin":
        return (
          <div className="space-y-4">
            <Input placeholder="Target Company Name (India or Global)" />
            <Textarea placeholder="Paste raw GSTIN/VAT records or unstructured supplier data..." className="min-h-[120px]" />
            <Button className="w-full">Extract and reconcile globally</Button>
            <Terminal className="mt-6" />
          </div>
        );
      case "audit":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Input placeholder="Action (e.g., READ, MUTATE)" />
              <Input placeholder="User ID" />
            </div>
            <Input placeholder="Resource URI" />
            <Textarea placeholder='{"reason": "support ticket #1234"}' className="min-h-[100px] font-mono text-xs" />
            <div className="grid grid-cols-2 gap-4">
              <Button className="w-full">Write audit entry</Button>
              <Button variant="outline" className="w-full">Verify chain</Button>
            </div>
            <Terminal className="mt-6" />
          </div>
        );
      default:
        return <div>Tool implementation pending.</div>;
    }
  };

  if (tool) {
    return (
      <div className="p-8 lg:p-12 max-w-5xl mx-auto min-h-screen bg-[#F9F8F4]">
        <div className="mb-8 flex items-center gap-4 text-sm text-slate-500">
          <Link to="/dashboard" className="hover:text-slate-900 transition-colors">Dashboard</Link>
          <span>/</span>
          <span>Live Tools</span>
          <span>/</span>
          <span className="text-slate-900 font-medium">{tool.title}</span>
        </div>

        <div className="mb-10">
          <div className="flex items-center gap-4 mb-4">
            <Badge variant="live" className="px-3 py-1 text-sm shadow-sm">LIVE TOOL</Badge>
            <span className="text-xs font-bold text-slate-400 tracking-widest uppercase">{tool.category}</span>
          </div>
          <h1 className="font-serif text-4xl lg:text-5xl font-bold text-slate-900 tracking-tight">{tool.title}</h1>
          <p className="text-lg text-slate-600 mt-4 max-w-2xl">{tool.description}</p>
        </div>

        <Card className="p-8 lg:p-10 shadow-lg border-slate-200/60 bg-white">
          {renderToolContent(tool.id)}
        </Card>
      </div>
    );
  }

  if (preview) {
    return (
      <div className="p-8 lg:p-12 max-w-4xl mx-auto min-h-screen bg-[#F9F8F4]">
        <div className="mb-8 flex items-center gap-4 text-sm text-slate-500">
          <Link to="/dashboard" className="hover:text-slate-900 transition-colors">Dashboard</Link>
          <span>/</span>
          <span>Platform Preview</span>
          <span>/</span>
          <span className="text-slate-900 font-medium">{preview.title}</span>
        </div>

        <div className="mb-10">
          <Badge variant="preview" className="mb-4 px-3 py-1 text-sm shadow-sm">PREVIEW / LAZY</Badge>
          <h1 className="font-serif text-4xl font-bold text-slate-900 tracking-tight">{preview.title}</h1>
          <p className="text-lg text-slate-600 mt-4">{preview.description}</p>
        </div>

        <Card className="p-12 text-center border-dashed border-2 border-slate-300 bg-slate-50/50">
          <svg className="w-16 h-16 mx-auto text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
          <h3 className="text-xl font-bold text-slate-800">Module Currently Lazy-Loaded</h3>
          <p className="text-slate-500 mt-2 max-w-md mx-auto">This module is part of the broader product suite roadmap and is not fully instantiated in the current browser runtime preview.</p>
          <Button variant="outline" className="mt-8" disabled>
            Request Access
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-12 flex flex-col items-center justify-center min-h-screen text-slate-500">
      <h2 className="text-2xl font-bold mb-2 text-slate-900">Feature not found</h2>
      <p>The requested feature module could not be located.</p>
      <Link to="/dashboard" className="mt-6 text-[#115E59] hover:underline font-medium">Return to Dashboard</Link>
    </div>
  );
}
