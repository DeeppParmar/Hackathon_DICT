import { Brain, Scan, Shield, Clock } from "lucide-react";

const stats = [
  { icon: Brain, label: "AI Powered", value: "Deep Learning" },
  { icon: Scan, label: "Accuracy", value: "95%+" },
  { icon: Clock, label: "Processing", value: "<3 Sec" },
  { icon: Shield, label: "Diseases", value: "3 Types" },
];

const HeroSection = () => {
  return (
    <section className="relative pt-32 pb-16">
      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center max-w-4xl mx-auto mb-16">


          <h1 className="font-display text-5xl md:text-6xl font-bold mb-6 tracking-tight">
            Advanced Medical Scan
            <span className="text-primary italic ml-3">Intelligence</span>
          </h1>

          <p className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Instant AI-powered detection for Tuberculosis, Pneumonia, and Bone Fractures with clinical-grade accuracy.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto animate-fade-in" style={{ animationDelay: '0.3s' }}>
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className="glass-card rounded-xl p-4 text-center group hover:border-primary/50 transition-all duration-300"
              style={{ animationDelay: `${0.4 + index * 0.1}s` }}
            >
              <div className="inline-flex p-2 rounded-lg bg-primary/10 mb-3 group-hover:bg-primary/20 transition-colors">
                <stat.icon className="w-5 h-5 text-primary" />
              </div>
              <div className="font-display font-bold text-lg">{stat.value}</div>
              <div className="text-xs text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
