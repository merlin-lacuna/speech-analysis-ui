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
        className="w-full h-4 accent-blue-500"
        readOnly
      />
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
  const audioRef = useRef<HTMLAudioElement>(null);

  // Mock data for the analysis results
  const confidenceAnalysis = {
    speechRate: "Very Low",
    pitchVariability: "Very Low",
    volumeConsistency: "Very Low",
    fillerWordUsage: "Very High",
    articulationClarity: "Very Low",
    strategicPausing: "Very Low"
  };

  const voiceMetrics = {
    amplitude: 30,
    pitch: 45,
    frequency: 60,
    energy: 25
  };

  const startRecording = async () => {
    try {
      // Clear previous results
      setSpectrogramUrl(null);
      setAnalysisResult(null);
      setAudioBlob(null);
      setAudioUrl(null);

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
      console.log("Audio analysis completed successfully:", data.message);
    } catch (error) {
      console.error("Error analyzing audio:", error);
      alert(`Error analyzing audio: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle audio time update
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
                    <div className="flex items-center space-x-2 bg-slate-50 p-3 rounded-md">
                      <button 
                        className="p-1.5 hover:bg-slate-200 rounded-full transition-colors"
                        onClick={() => audioRef.current?.play()}
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="20"
                          height="20"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="text-slate-700"
                        >
                          <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                      </button>
                      <div className="text-xs text-slate-500 w-16 flex-shrink-0">
                        {`${Math.floor(currentTime / 60)}:${String(Math.floor(currentTime % 60)).padStart(2, '0')} / ${Math.floor(duration / 60)}:${String(Math.floor(duration % 60)).padStart(2, '0')}`}
                      </div>
                      <div className="flex-1">
                        <input
                          type="range"
                          min="0"
                          max={duration || 100}
                          value={currentTime}
                          onChange={(e) => {
                            const time = parseFloat(e.target.value);
                            setCurrentTime(time);
                            if (audioRef.current) audioRef.current.currentTime = time;
                          }}
                          className="w-full accent-blue-500"
                        />
                      </div>
                      <button className="p-1.5 hover:bg-slate-200 rounded-full transition-colors">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="20"
                          height="20"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="text-slate-700"
                        >
                          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                        </svg>
                      </button>
                      <audio
                        ref={audioRef}
                        src={audioUrl}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleTimeUpdate}
                        className="hidden"
                        controls
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
            <Card className="shadow-md bg-[#1e293b] text-white">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">Voice Analysis</h3>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Confidence Analysis</span>
                    <span className="text-xs text-red-300">{confidenceAnalysis.speechRate}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Pitch Variability</span>
                    <span className="text-xs text-red-300">{confidenceAnalysis.pitchVariability}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Volume Consistency</span>
                    <span className="text-xs text-red-300">{confidenceAnalysis.volumeConsistency}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Filler Word Usage</span>
                    <span className="text-xs text-blue-300">{confidenceAnalysis.fillerWordUsage}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "100%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Articulation Clarity</span>
                    <span className="text-xs text-red-300">{confidenceAnalysis.articulationClarity}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
                    <div className="h-full bg-red-500 rounded-full" style={{ width: "10%" }}></div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm">Strategic Pausing</span>
                    <span className="text-xs text-red-300">{confidenceAnalysis.strategicPausing}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full w-full overflow-hidden">
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
            <Card className="shadow-md bg-[#1e293b] text-white">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-6">Voice Metrics</h3>

                <div className="grid grid-cols-2 gap-6">
                  {analysisResult && analysisResult.features ? (
                    <>
                      {Object.entries(analysisResult.features).slice(0, 4).map(([key, value]) => (
                        <GaugeChart 
                          key={key} 
                          label={key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')} 
                          value={Math.round(value * 100)} 
                        />
                      ))}
                    </>
                  ) : (
                    <>
                      <GaugeChart label="Amplitude" value={voiceMetrics.amplitude} />
                      <GaugeChart label="Pitch" value={voiceMetrics.pitch} />
                      <GaugeChart label="Frequency Velocity" value={voiceMetrics.frequency} />
                      <GaugeChart label="Energy" value={voiceMetrics.energy} />
                    </>
                  )}
                </div>
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
                <h3 className="text-base font-semibold mb-4">Our confidence analysis evaluates several key aspects of your speech:</h3>

                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li>- Speech Rate: A steady, moderate pace is often associated with confidence.</li>
                  <li>- Pitch Variability: Controlled variation in pitch can indicate assurance.</li>
                  <li>- Volume Consistency: Maintaining a steady volume suggests self-assurance.</li>
                  <li>- Filler Word Usage: Fewer filler words often indicate more confident speech.</li>
                  <li>- Articulation Clarity: Clear pronunciation is a sign of confidence.</li>
                  <li>- Strategic Pausing: Thoughtful use of pauses can demonstrate composure.</li>
                </ul>

                <p className="mt-4 text-sm text-muted-foreground">These metrics work together to provide an overall assessment of perceived confidence in your speech patterns.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </div>
  )
}

