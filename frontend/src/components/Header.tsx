import { Activity, Zap } from "lucide-react";

const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-border/30">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/5 border border-primary/20">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-display text-lg font-bold tracking-tight">
                MediScan <span className="text-primary">AI</span>
              </h1>
              <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-medium">Early Disease Detection</p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground">
              <a href="#analyze" className="hover:text-primary transition-colors">Analyze</a>
              <a href="#features" className="hover:text-primary transition-colors">Features</a>
            </div>
            <div className="h-4 w-px bg-border/50 hidden md:block" />
            <div className="flex items-center gap-2">
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
