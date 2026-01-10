import { Cpu, Eye, Zap, Shield, Layers, Lightbulb } from "lucide-react";

const features = [
  {
    icon: Cpu,
    title: "AI Analysis",
    description: "Deep learning models for precise feature extraction across medical imaging.",
  },
  {
    icon: Lightbulb,
    title: "Explainable AI",
    description: "Visualization shows exactly why the model made its decision.",
  },
  {
    icon: Zap,
    title: "Fast Results",
    description: "Get analysis in seconds with our optimized inference pipeline.",
  },
  {
    icon: Shield,
    title: "Multi-Disease",
    description: "Detect Tuberculosis, Pneumonia, and Bone Abnormalities in one platform.",
  },
];

const FeaturesSection = () => {
  return (
    <section className="py-20 border-t border-border/50" id="features">
      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4 tracking-tight">
            Integrated <span className="text-primary">Intelligence</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            Our platform utilizes standardized deep learning architectures for clinical diagnostic support.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="glass-card rounded-2xl p-6 group hover:border-primary/50 transition-all duration-300 hover:-translate-y-1"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="p-3 rounded-xl bg-primary/10 w-fit mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-display font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
