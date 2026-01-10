import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import AnalysisSection from "@/components/AnalysisSection";
import FeaturesSection from "@/components/FeaturesSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <AnalysisSection />
        <FeaturesSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
