"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Mic, Square, Loader2, Check } from "lucide-react"
import { Label } from "@/components/ui/label"

export default function SampleIngestion() {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [label, setLabel] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [audioChunks, setAudioChunks] = useState<Blob[]>([])
  const [ingestionSuccess, setIngestionSuccess] = useState(false)

  const startRecording = async () => {
    try {
      // Clear previous state
      setAudioBlob(null)
      setIngestionSuccess(false)

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

  const ingestSample = async () => {
    if (!audioBlob || !label.trim()) return

    setIsProcessing(true)
    setIngestionSuccess(false)
    console.log("Audio blob size:", audioBlob.size, "bytes")
    console.log("Audio blob type:", audioBlob.type)

    try {
      const formData = new FormData()
      formData.append("audio", audioBlob)
      formData.append("label", label.trim())
      console.log("Sending request to ingest audio sample...")

      const response = await fetch("/api/ingest-sample", {
        method: "POST",
        body: formData,
      })

      console.log("Response status:", response.status)
      const data = await response.json()
      console.log("Response data:", data)

      if (!response.ok) {
        throw new Error(data.error || "Failed to ingest audio sample")
      }

      setIngestionSuccess(true)
      setLabel("")  // Clear the label for the next recording
      console.log("Sample ingestion completed successfully")
    } catch (error) {
      console.error("Error ingesting sample:", error)
      alert(`Error ingesting sample: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container flex items-center justify-center min-h-screen py-12">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Ingest Speech Sample</CardTitle>
          <CardDescription>Record and label speech samples for the emotion database</CardDescription>
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
              
              <div className="space-y-2">
                <Label htmlFor="label">Emotion Label</Label>
                <Input
                  id="label"
                  placeholder="Enter emotion label (e.g., happy, sad, angry)"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                />
              </div>

              <Button 
                className="w-full" 
                onClick={ingestSample} 
                disabled={isProcessing || !label.trim()}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Ingesting...
                  </>
                ) : (
                  "Ingest Sample"
                )}
              </Button>
            </div>
          )}

          {ingestionSuccess && (
            <div className="flex items-center justify-center p-4 text-green-600 bg-green-50 rounded-md">
              <Check className="w-5 h-5 mr-2" />
              <span>Sample successfully ingested!</span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 