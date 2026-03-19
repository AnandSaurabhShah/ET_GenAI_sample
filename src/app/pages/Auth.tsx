import React from 'react';
import { useNavigate } from 'react-router';
import { Button, Input, Card, cn } from '../components/ui';

export function Auth() {
  const navigate = useNavigate();

  const handleSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/dashboard');
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F9F8F4] p-4 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 rounded-full bg-[#115E59]/5 blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-80 h-80 rounded-full bg-orange-500/5 blur-3xl pointer-events-none"></div>

      <Card className="w-full max-w-md p-8 shadow-2xl relative z-10 border-slate-200/60">
        <div className="mb-6 flex justify-center">
          <div className="px-3 py-1 bg-slate-100 border border-slate-200 rounded-full text-[10px] font-bold tracking-widest text-slate-500 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
            ET GENAI HACKATHON SUBMISSION
          </div>
        </div>

        <div className="mb-8 text-center">
          <h1 className="font-serif text-3xl font-bold text-slate-900">Sign In</h1>
          <p className="mt-3 text-sm text-slate-600 font-medium">Bharat's Enterprise AI for the Global Fortune 500</p>
        </div>

        <form onSubmit={handleSignIn} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-700" htmlFor="email">
              User ID / Email
            </label>
            <Input id="email" type="email" placeholder="name@company.com" required />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-700" htmlFor="password">
              Password
            </label>
            <Input id="password" type="password" required />
          </div>

          <Button type="submit" className="w-full">
            Sign In
          </Button>
        </form>

        <div className="mt-6 flex items-center justify-center space-x-2 text-sm text-slate-500">
          <span className="h-px w-full bg-slate-200"></span>
          <span className="shrink-0 px-2">or continue with</span>
          <span className="h-px w-full bg-slate-200"></span>
        </div>

        <div className="mt-6 space-y-3">
          <Button variant="outline" className="w-full" type="button">
            <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
              <path d="M1 1h22v22H1z" fill="none" />
            </svg>
            Sign in with Google
          </Button>
          <Button variant="outline" className="w-full" type="button">
             <svg className="mr-2 h-4 w-4" viewBox="0 0 21 21">
               <path fill="#f25022" d="M1 1h9v9H1z"/>
               <path fill="#00a4ef" d="M1 11h9v9H1z"/>
               <path fill="#7fba00" d="M11 1h9v9h-9z"/>
               <path fill="#ffb900" d="M11 11h9v9h-9z"/>
             </svg>
            Sign in with Microsoft
          </Button>
        </div>
      </Card>
    </div>
  );
}
