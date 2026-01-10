import { useEffect, useState } from "react";
import { Sparkles, RotateCcw } from "lucide-react";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import ImageUpload from "./ImageUpload";
import AnalysisResults, { AnalysisResult } from "./AnalysisResults";
import { API_ENDPOINTS } from "@/config";

const AnalysisSection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [scanType, setScanType] = useState<'auto' | 'chest' | 'bone' | 'ct'>('auto');

  const runAnalysis = async (file: File) => {
    setIsAnalyzing(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      if (scanType !== 'auto') formData.append('scan_type', scanType);

      const name = file.name.toLowerCase();
      const isDicom = name.endsWith('.dcm');
      const boneKeywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura'];
      const isBone = boneKeywords.some((k) => name.includes(k));
      if (isDicom) formData.append('model', 'rsna');
      else if (isBone) formData.append('model', 'mura');

      const response = await fetch(API_ENDPOINTS.ANALYZE, {
        method: 'POST',
        body: formData
      });

      const modelUsed = response.headers.get('X-Model-Used');
      if (modelUsed) console.log('Model used:', modelUsed);

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
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setResults(null);
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
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    await runAnalysis(selectedImage);
  };

  return (
    <section className="py-16" id="analyze">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
            Start <span className="gradient-text">Analysis</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload A Chest X-Ray, CT Scan, Or Radiograph For AI-Powered Disease Detection
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Upload Section */}
          <div className="space-y-6">
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              onClear={handleClear}
            />

            <div className="grid gap-2">
              <div className="text-sm text-muted-foreground">Scan Type</div>
              <Select value={scanType} onValueChange={(v) => setScanType(v as any)}>
                <SelectTrigger>
                  <SelectValue placeholder="Auto" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="chest">Chest X-Ray</SelectItem>
                  <SelectItem value="bone">Bone X-Ray</SelectItem>
                  <SelectItem value="ct">CT / DICOM</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
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

          {/* Results Section */}
          <div>
            <AnalysisResults results={results} isAnalyzing={isAnalyzing} />
          </div>
        </div>
      </div>
    </section>
  );
};

export default AnalysisSection;
