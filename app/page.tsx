"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Mic, Square, Loader2 } from "lucide-react"
import Image from "next/image"

interface EmotionAnalysis {
  emotion: string
  confidence: number
  features: Record<string, number>
}

export default function AudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [spectrogramUrl, setSpectrogramUrl] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<EmotionAnalysis | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [audioChunks, setAudioChunks] = useState<Blob[]>([])

  const startRecording = async () => {
    try {
      // Clear previous results
      setSpectrogramUrl(null)
      setAnalysisResult(null)
      setAudioBlob(null)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/wav'  // Try WAV format first
      })
      setMediaRecorder(recorder)

      const chunks: Blob[] = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data)
        }
      }

      recorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: "audio/wav" })
        setAudioBlob(audioBlob)
        setAudioChunks(chunks)
      }

      recorder.start()
      setIsRecording(true)
    } catch (error) {
      // If WAV is not supported, fallback to PCM
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=pcm'
        })
        setMediaRecorder(recorder)

        const chunks: Blob[] = []
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data)
          }
        }

        recorder.onstop = () => {
          const audioBlob = new Blob(chunks, { type: "audio/webm" })
          setAudioBlob(audioBlob)
          setAudioChunks(chunks)
        }

        recorder.start()
        setIsRecording(true)
      } catch (error) {
        console.error("Error accessing microphone:", error)
        alert("Error accessing microphone. Please ensure you have granted permission.")
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop()
      // Stop all audio tracks
      mediaRecorder.stream.getTracks().forEach((track) => track.stop())
      setIsRecording(false)
    }
  }

  const analyzeAudio = async () => {
    if (!audioBlob) return

    setIsProcessing(true)
    console.log("Audio blob size:", audioBlob.size, "bytes")
    console.log("Audio blob type:", audioBlob.type)

    try {
      const formData = new FormData()
      formData.append("audio", audioBlob)
      console.log("Sending request to analyze audio...")

      const response = await fetch("/api/generate-spectrogram", {
        method: "POST",
        body: formData,
      })

      console.log("Response status:", response.status)
      const data = await response.json()
      console.log("Response data:", data)

      if (!response.ok) {
        throw new Error(data.error || "Failed to analyze audio")
      }

      setSpectrogramUrl(data.spectrogramUrl)
      setAnalysisResult({
        emotion: data.emotion,
        confidence: data.confidence,
        features: data.features
      })
      console.log("Audio analysis completed successfully:", data.message)
    } catch (error) {
      console.error("Error analyzing audio:", error)
      alert(`Error analyzing audio: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const getEmotionColor = (emotion: string) => {
    const colors = {
      happy: "text-yellow-500",
      sad: "text-blue-500",
      angry: "text-red-500",
      neutral: "text-gray-500"
    }
    return colors[emotion as keyof typeof colors] || "text-gray-500"
  }

  return (
    <div className="container flex items-center justify-center min-h-screen py-12">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Speech Emotion Analyzer</CardTitle>
          <CardDescription>Record your voice to analyze its emotional content</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
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

          {audioBlob && !isRecording && (
            <div className="space-y-4">
              <div className="flex justify-center">
                <audio controls src={URL.createObjectURL(audioBlob)} className="w-full" />
              </div>
              <Button className="w-full" onClick={analyzeAudio} disabled={isProcessing}>
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

          {analysisResult && (
            <div className="mt-6 space-y-4">
              <div className="text-center">
                <h3 className="text-2xl font-bold text-blue-600">
                  {analysisResult.emotion.charAt(0).toUpperCase() + analysisResult.emotion.slice(1)}
                </h3>
                <p className="text-sm text-gray-500">
                  Match Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="overflow-hidden border rounded-md">
                <Image
                  src={spectrogramUrl || "/placeholder.svg"}
                  alt="Audio Spectrogram"
                  width={400}
                  height={300}
                  className="w-full"
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

