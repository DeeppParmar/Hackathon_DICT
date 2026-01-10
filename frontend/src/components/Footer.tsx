import { Activity, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 border-t border-border/30">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-secondary border border-border">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="font-display font-bold text-sm">
                MediScan <span className="text-primary">AI</span>
              </p>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Clinical Image Analysis</p>
            </div>
          </div>

          <div className="text-center md:text-left">
            <p className="text-xs text-muted-foreground">
              Clinical Image Analysis Platform
            </p>
          </div>

          <div className="text-center md:text-right">
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-semibold">
              © 2026 MediScan AI
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
