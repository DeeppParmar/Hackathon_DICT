import { useEffect, useState } from "react";
import { Sparkles, RotateCcw, Stethoscope, Bone, FlaskConical, Wand2, Cpu, Clock } from "lucide-react";
import { Button } from "./ui/button";
import ImageUpload from "./ImageUpload";
import AnalysisResults, { AnalysisResult } from "./AnalysisResults";
import { API_ENDPOINTS } from "@/config";

type ScanType = 'auto' | 'chest' | 'bone' | 'tb';

interface ScanTypeOption {
  id: ScanType;
  label: string;
  description: string;
  icon: React.ElementType;
  model: string;
  color: string;
}

const scanTypeOptions: ScanTypeOption[] = [
  {
    id: 'auto',
    label: 'Auto Detect',
    description: 'AI chooses the best model',
    icon: Wand2,
    model: 'Automatic',
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'chest',
    label: 'Chest X-Ray',
    description: '14 disease detection',
    icon: Stethoscope,
    model: 'CheXNet',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'bone',
    label: 'Bone X-Ray',
    description: 'Musculoskeletal analysis',
    icon: Bone,
    model: 'MURA',
    color: 'from-orange-500 to-amber-500'
  },
  {
    id: 'tb',
    label: 'TB Screening',
    description: 'Tuberculosis detection',
    icon: FlaskConical,
    model: 'TuberculosisNet',
    color: 'from-green-500 to-emerald-500'
  }
];

const AnalysisSection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [scanType, setScanType] = useState<ScanType>('auto');
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const runAnalysis = async (file: File) => {
    setIsAnalyzing(true);
    setResults(null);
    setModelUsed(null);
    setProcessingTime(null);

    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('image', file);

      // Map scan type to backend expected values
      if (scanType === 'tb') {
        formData.append('scan_type', 'chest');
        formData.append('model', 'tuberculosis');
      } else if (scanType !== 'auto') {
        formData.append('scan_type', scanType);
      }

      const name = file.name.toLowerCase();
      const isDicom = name.endsWith('.dcm');
      const boneKeywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura'];
      const isBone = boneKeywords.some((k) => name.includes(k));
      if (isDicom) formData.append('model', 'rsna');
      else if (isBone && scanType === 'auto') formData.append('model', 'mura');

      const response = await fetch(API_ENDPOINTS.ANALYZE, {
        method: 'POST',
        body: formData
      });

      const endTime = Date.now();
      setProcessingTime(endTime - startTime);

      // Get model used from response header
      const usedModel = response.headers.get('X-Model-Used');
      if (usedModel) {
        const modelNames: Record<string, string> = {
          'chexnet': 'CheXNet',
          'mura': 'MURA',
          'tuberculosis': 'TuberculosisNet',
          'rsna': 'RSNA',
          'unet': 'UNet'
        };
        setModelUsed(modelNames[usedModel] || usedModel);
      }

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        if (Array.isArray(data)) {
          setResults(data);
          return;
        }
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      if (Array.isArray(data)) {
        setResults(data);
      } else {
        console.error('Unexpected API response format:', data);
        throw new Error('Invalid response format from API');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setResults([{
        disease: 'Connection Error',
        confidence: 0,
        status: 'warning',
        description: `Could not connect to backend. Make sure the server is running at ${API_ENDPOINTS.ANALYZE}`
      }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setResults(null);
    void runAnalysis(file);
  };

  useEffect(() => {
    if (!selectedImage) return;
    if (isAnalyzing) return;
    void runAnalysis(selectedImage);
  }, [scanType]);

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setIsAnalyzing(false);
    setModelUsed(null);
    setProcessingTime(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    await runAnalysis(selectedImage);
  };

  const selectedOption = scanTypeOptions.find(o => o.id === scanType)!;

  return (
    <section className="py-16" id="analyze">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
            AI-Powered <span className="gradient-text">Disease Detection</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload a medical scan and select the analysis type for instant AI-powered diagnosis
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto animate-fade-in" style={{ animationDelay: '0.3s' }}>          {scanTypeOptions.map((option) => {
          const Icon = option.icon;
          const isSelected = scanType === option.id;
          return (
            <button
              key={option.id}
              onClick={() => setScanType(option.id)}
              disabled={isAnalyzing}
              className={`relative p-5 rounded-lg border transition-all duration-200 text-left
                  ${isSelected
                  ? 'border-primary bg-primary/5 ring-1 ring-primary/20'
                  : 'border-border bg-card'
                }
                  ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
            >
              <div className={`w-12 h-12 rounded-full bg-secondary flex items-center justify-center mb-4 border border-border
                  ${isSelected ? 'bg-primary/10 border-primary/20' : ''}`}>
                <Icon className={`w-5 h-5 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
              </div>
              <h3 className="font-bold text-sm mb-1">{option.label}</h3>
              <p className="text-[11px] text-muted-foreground leading-tight">{option.description}</p>
            </button>
          );
        })}
        </div>

        {(modelUsed || isAnalyzing) && (<div className="flex items-center justify-center gap-4 mb-6">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border">
            <Cpu className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">
              {isAnalyzing ? 'Processing...' : `Model: ${modelUsed}`}
            </span>
          </div>
          {processingTime && !isAnalyzing && (
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border">
              <Clock className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {(processingTime / 1000).toFixed(2)}s
              </span>
            </div>
          )}
        </div>
        )}

        <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          <div className="space-y-6">            <ImageUpload
            onImageSelect={handleImageSelect}
            selectedImage={selectedImage}
            onClear={handleClear}
          />

            <div className="flex gap-4">
              <Button
                variant="glow"
                size="lg"
                className="flex-1"
                onClick={handleAnalyze}
                disabled={!selectedImage || isAnalyzing}
              >
                <Sparkles className="w-5 h-5" />
                {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
              </Button>

              {(selectedImage || results) && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleClear}
                  disabled={isAnalyzing}
                >
                  <RotateCcw className="w-5 h-5" />
                  Reset
                </Button>
              )}
            </div>
          </div>

          <div>            <AnalysisResults results={results} isAnalyzing={isAnalyzing} modelUsed={modelUsed} />
          </div>
        </div>
      </div>
    </section>
  );
};

export default AnalysisSection;

