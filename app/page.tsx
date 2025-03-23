"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Mic, Square, Loader2 } from "lucide-react"

// Custom component for the voice metrics gauge charts
const GaugeChart = ({ label, value }: { label: string, value: number }) => {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs font-medium text-blue-400">{value}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        className="gauge-slider w-full h-4 accent-blue-500"
        readOnly
      />
      <style jsx>{`
        .gauge-slider {
          /* Ensure these styles are scoped only to this component */
          accent-color: #3b82f6;
        }
      `}</style>
    </div>
  );
};

interface EmotionAnalysis {
  emotion: string
  confidence: number
  features: Record<string, number>
}

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [spectrogramUrl, setSpectrogramUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<EmotionAnalysis | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [hasAnalyzed, setHasAnalyzed] = useState(false); // Track if analysis has happened
  const audioRef = useRef<HTMLAudioElement>(null);

  // Default metrics before analysis
  const confidenceAnalysis = {
    speechRate: "Very Low",
    pitchVariability: "Very Low",
    volumeConsistency: "Very Low",
    fillerWordUsage: "Very High",
    articulationClarity: "Very Low",
    strategicPausing: "Very Low"
  };

  // Default metrics before analysis (show empty values)
  const voiceMetrics = {
    tonalShape: 0,
    vocalEnergy: 0,
    pitchVariation: 0,
    vocalClarity: 0
  };

  const startRecording = async () => {
    try {
      // Clear previous results
      setSpectrogramUrl(null);
      setAnalysisResult(null);
      setAudioBlob(null);
      setAudioUrl(null);
      setHasAnalyzed(false); // Reset analysis state

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/wav'  // Try WAV format first
      });
      setMediaRecorder(recorder);

      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: "audio/wav" });
        setAudioBlob(audioBlob);
        setAudioChunks(chunks);
        setAudioUrl(URL.createObjectURL(audioBlob));
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      // If WAV is not supported, fallback to PCM
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=pcm'
        });
        setMediaRecorder(recorder);

        const chunks: Blob[] = [];
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        recorder.onstop = () => {
          const audioBlob = new Blob(chunks, { type: "audio/webm" });
          setAudioBlob(audioBlob);
          setAudioChunks(chunks);
          setAudioUrl(URL.createObjectURL(audioBlob));
        };

        recorder.start();
        setIsRecording(true);
      } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Error accessing microphone. Please ensure you have granted permission.");
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      // Stop all audio tracks
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  };

  const analyzeEmotion = async () => {
    if (!audioBlob) return;

    setIsProcessing(true);
    console.log("Audio blob size:", audioBlob.size, "bytes");
    console.log("Audio blob type:", audioBlob.type);

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob);
      console.log("Sending request to analyze audio...");

      const response = await fetch("/api/generate-spectrogram", {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status);
      const data = await response.json();
      console.log("Response data:", data);

      if (!response.ok) {
        throw new Error(data.error || "Failed to analyze audio");
      }

      setSpectrogramUrl(data.spectrogramUrl);
      setAnalysisResult({
        emotion: data.emotion,
        confidence: data.confidence,
        features: data.features
      });
      setHasAnalyzed(true); // Mark that analysis has happened
      console.log("Audio analysis completed successfully:", data.message);
    } catch (error) {
      console.error("Error analyzing audio:", error);
      alert(`Error analyzing audio: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle audio time update - keeping this for compatibility
  // but we're now using the native audio controls
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration || 0);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* HEADER */}
      <header className="p-4">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }} className="flex items-center">
            <div className="flex-shrink-0" style={{ height: "12vh" }}>
              <video
                src="/animations/jumping_mushroom2.webm"
                autoPlay
                loop
                muted
                className="h-full w-auto object-contain"
              />
            </div>
            <div className="flex-1 pl-4">
              <div className="bg-pink-100 p-4 rounded-lg shadow-sm">
                <p className="text-base font-medium text-gray-800">Can you teach me how to speak like a CEO?</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* SECTION 1: Speech Emotion Analyzer */}
      <section className="py-8">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }}>
            <Card className="shadow-md">
              <CardContent className="p-6">
                <h2 className="text-xl font-semibold text-center mb-2">Speech Emotion Analyzer</h2>
                <p className="text-center text-muted-foreground text-sm mb-6">Record your voice to analyze its emotional content</p>

                <div className="flex flex-col items-center justify-center gap-4">
                  {isRecording ? (
                    <Button variant="destructive" size="lg" className="w-16 h-16 rounded-full" onClick={stopRecording}>
                      <Square className="w-6 h-6" />
                    </Button>
                  ) : (
                    <Button
                      variant="default"
                      size="lg"
                      className="w-16 h-16 rounded-full bg-red-500 hover:bg-red-600"
                      onClick={startRecording}
                    >
                      <Mic className="w-6 h-6" />
                    </Button>
                  )}
                  <div className="text-center">
                    {isRecording ? (
                      <p className="text-sm font-medium text-red-500">Recording... Click to stop</p>
                    ) : (
                      <p className="text-sm font-medium">Click to start recording</p>
                    )}
                  </div>
                </div>

                {audioUrl && !isRecording && (
                  <div className="mt-6 space-y-4">
                    <div className="bg-slate-50 p-3 rounded-md flex justify-center">
                      <style jsx>{`
                        /* Override global styles for the audio element within this component */
                        audio {
                          /* Reset any inherited styles */
                          width: 100%;
                          accent-color: initial;
                        }
                        audio::-webkit-media-controls-panel {
                          background-color: #f8fafc;
                        }
                        audio::-webkit-media-controls-current-time-display,
                        audio::-webkit-media-controls-time-remaining-display {
                          color: #334155;
                        }
                      `}</style>
                      <audio
                        ref={audioRef}
                        src={audioUrl}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleTimeUpdate}
                        controls
                        className="w-full"
                      />
                    </div>
                    <Button className="w-full" onClick={analyzeEmotion} disabled={isProcessing}>
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        "Analyze Emotion"
                      )}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* SECTION 2: Voice Analysis */}
      <section className="py-8">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }}>
            <Card className="shadow-md">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">Voice Analysis</h3>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Confidence Analysis</span>
                    <span className="text-xs text-red-500">{confidenceAnalysis.speechRate}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Pitch Variability</span>
                    <span className="text-xs text-red-500">{confidenceAnalysis.pitchVariability}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Volume Consistency</span>
                    <span className="text-xs text-red-500">{confidenceAnalysis.volumeConsistency}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Filler Word Usage</span>
                    <span className="text-xs text-blue-500">{confidenceAnalysis.fillerWordUsage}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "100%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Articulation Clarity</span>
                    <span className="text-xs text-red-500">{confidenceAnalysis.articulationClarity}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Strategic Pausing</span>
                    <span className="text-xs text-red-500">{confidenceAnalysis.strategicPausing}</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* SECTION 3: Voice Metrics */}
      <section className="py-8">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }}>
            <Card className="shadow-md">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-6">Voice Metrics</h3>

                {!hasAnalyzed ? (
                  <div className="text-center py-6">
                    <p className="text-sm text-gray-500">Record your voice and click "Analyze Emotion" to see your voice metrics</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-6">
                    {analysisResult && analysisResult.features ? (
                      <>
                        {/* Map feature indices to human-readable names with better normalization */}
                        <GaugeChart 
                          key="tonal_shape" 
                          label="Tonal Shape" 
                          value={Math.min(100, Math.max(0, Math.round(
                            (Math.abs(analysisResult.features.feature_0) / 500) * 100
                          )))} 
                        />
                        <GaugeChart 
                          key="vocal_energy" 
                          label="Vocal Energy" 
                          value={Math.min(100, Math.max(0, Math.round(
                            (Math.abs(analysisResult.features.feature_1) / 150) * 100
                          )))} 
                        />
                        <GaugeChart 
                          key="pitch_variation" 
                          label="Pitch Variation" 
                          value={Math.min(100, Math.max(0, Math.round(
                            (analysisResult.features.feature_22 / 60) * 100
                          )))} 
                        />
                        <GaugeChart 
                          key="vocal_clarity" 
                          label="Vocal Clarity" 
                          value={Math.min(100, Math.max(0, Math.round(
                            (analysisResult.features.feature_20 / 150) * 100
                          )))} 
                        />
                      </>
                    ) : null}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* SECTION 4: Spectrogram and Emotion */}
      <section className="py-8">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }}>
            <Card className="shadow-md">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">Spectrogram</h3>
                
                {!hasAnalyzed ? (
                  <div className="text-center py-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
                    <p className="text-sm text-gray-500">Record your voice and click "Analyze Emotion" to see your audio spectrogram</p>
                  </div>
                ) : (
                  <>
                    <div className="bg-slate-50 p-4 rounded-lg border border-slate-100 flex justify-center">
                      <img
                        src={spectrogramUrl || "/placeholder.svg"}
                        alt="Audio Spectrogram"
                        style={{ width: 'auto', height: 'auto', maxHeight: '200px' }}
                        className="rounded-md shadow-sm"
                      />
                    </div>
                    
                    {analysisResult && (
                      <div className="mt-4 text-center">
                        <h3 className="text-xl font-bold text-blue-600">
                          {analysisResult.emotion.charAt(0).toUpperCase() + analysisResult.emotion.slice(1)}
                        </h3>
                        <p className="text-sm text-gray-500">
                          Match Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* SECTION 5: Informational Panel */}
      <section className="py-8">
        <div className="container mx-auto">
          <div style={{ width: '40%', maxWidth: '600px', margin: '0 auto' }}>
            <Card className="shadow-md">
              <CardContent className="p-6">
                <h3 className="text-base font-semibold mb-4">Our voice metrics evaluate several key aspects of your speech:</h3>

                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li>- <strong>Tonal Shape:</strong> Represents the overall contour of your speech sounds and how they form your unique voice signature.</li>
                  <li>- <strong>Vocal Energy:</strong> Measures the power and projection in your voice, indicating how effectively your voice carries.</li>
                  <li>- <strong>Pitch Variation:</strong> Tracks how your voice modulates between higher and lower tones, which conveys emotion and emphasis.</li>
                  <li>- <strong>Vocal Clarity:</strong> Evaluates the crispness and definition in your speech, affecting how clearly your words are understood.</li>
                </ul>

                <p className="mt-4 text-sm text-muted-foreground">These metrics are derived from Mel-frequency cepstral coefficients (MFCCs), which are specialized measurements used in speech analysis to capture the unique characteristics of human voice.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </div>
  )
}

