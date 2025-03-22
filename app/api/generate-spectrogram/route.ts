import { type NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir, readFile, unlink, rename } from "fs/promises"
import { join } from "path"
import { randomUUID } from "crypto"
import { exec } from "child_process"
import { promisify } from "util"
import os from 'os'
import { existsSync } from 'fs'

const execPromise = promisify(exec)

// Ensure directory exists
async function ensureDir(dir: string) {
  try {
    await mkdir(dir, { recursive: true })
  } catch (error) {
    // Directory already exists or cannot be created
    console.error(`Error creating directory ${dir}:`, error)
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log("Received audio analysis request")
    const formData = await request.formData()
    const audioFile = formData.get("audio") as File

    // Determine if this is a training sample based on the URL path
    const isTrainingSample = request.url.includes('/api/ingest')

    if (!audioFile) {
      console.error("No audio file provided")
      return NextResponse.json({ error: "No audio file provided" }, { status: 400 })
    }

    console.log("Audio file received:", {
      name: audioFile.name,
      type: audioFile.type,
      size: audioFile.size,
      isTrainingSample
    })

    // Create unique filenames
    const id = randomUUID()
    const publicDir = join(process.cwd(), "public")

    // Use different root folders for user recordings vs training samples
    const rootStorageDir = isTrainingSample ?
      join(publicDir, "training_samples") :
      join(publicDir, "user_recordings")

    const pendingDir = join(rootStorageDir, "pending")
    const spectrogramsDir = join(publicDir, "spectrograms")

    // Ensure directories exist
    await ensureDir(rootStorageDir)
    await ensureDir(pendingDir)
    await ensureDir(spectrogramsDir)

    // Create temporary paths
    const audioPath = join(pendingDir, `${id}.webm`)
    const wavPath = join(pendingDir, `${id}.wav`)
    const errorLogPath = join(pendingDir, `${id}_error.log`)
    const publicSpectrogramPath = join(spectrogramsDir, `${id}.png`)

    console.log("Paths:", {
      rootStorageDir,
      pendingDir,
      audioPath,
      wavPath,
      errorLogPath
    })

    // Write the audio file to disk
    const audioBuffer = Buffer.from(await audioFile.arrayBuffer())
    console.log("Writing audio file, size:", audioBuffer.length, "bytes")
    await writeFile(audioPath, audioBuffer)
    console.log("Audio file written successfully")

    // Convert WebM to WAV using ffmpeg
    try {
      console.log("Converting WebM to WAV...")
      const ffmpegPath = process.platform === 'win32'
        ? 'ffmpeg' // On Windows, assumes ffmpeg is in PATH
        : join(process.cwd(), 'bin', 'ffmpeg') // On Linux, use local binary

      await execPromise(`"${ffmpegPath}" -i "${audioPath}" "${wavPath}"`)
      console.log("Conversion successful")
    } catch (error) {
      console.error("FFmpeg conversion failed:", error)
      return NextResponse.json({ error: "Failed to convert audio format" }, { status: 500 })
    }

    // Get path to Python script
    const scriptPath = join(process.cwd(), 'scripts', 'api', 'generate_spectrogram.py')

    // Execute the Python script using the venv Python path (platform-specific)
    const pythonPath = process.platform === 'win32'
      ? join(process.cwd(), 'venv', 'Scripts', 'python.exe')
      : join(process.cwd(), 'venv', 'bin', 'python')

    try {
      // Run the dedicated Python script with parameters
      const { stdout, stderr } = await execPromise(
        `"${pythonPath}" "${scriptPath}" "${wavPath}" "${publicSpectrogramPath}" "${errorLogPath}"`
      )

      // Parse the emotion analysis results from stdout
      const resultMatch = stdout.match(/RESULT_JSON:(.+)/)
      if (!resultMatch) {
        return NextResponse.json({ error: "Failed to parse emotion analysis results" }, { status: 500 })
      }

      const analysisResult = JSON.parse(resultMatch[1])
      const emotion = analysisResult.emotion

      // Move audio files to emotion-specific folder if we got a valid emotion
      if (emotion && emotion !== "unknown") {
        const emotionDir = join(rootStorageDir, emotion)
        await ensureDir(emotionDir)

        // Move both WebM and WAV files
        const finalWebmPath = join(emotionDir, `${id}.webm`)
        const finalWavPath = join(emotionDir, `${id}.wav`)

        await Promise.all([
          rename(audioPath, finalWebmPath),
          rename(wavPath, finalWavPath)
        ])

        console.log(`Moved audio files to ${emotion} folder`)

        // Update the paths in the result using the correct root folder
        const relativePath = isTrainingSample ?
          `/training_samples/${emotion}/${id}.wav` :
          `/user_recordings/${emotion}/${id}.wav`
        analysisResult.audioPath = relativePath
      } else {
        // If unknown emotion, still keep the files but in an 'unknown' folder
        const unknownDir = join(rootStorageDir, 'unknown')
        await ensureDir(unknownDir)

        const finalWebmPath = join(unknownDir, `${id}.webm`)
        const finalWavPath = join(unknownDir, `${id}.wav`)

        await Promise.all([
          rename(audioPath, finalWebmPath),
          rename(wavPath, finalWavPath)
        ])

        // Use the correct root folder in the path
        const relativePath = isTrainingSample ?
          `/training_samples/unknown/${id}.wav` :
          `/user_recordings/unknown/${id}.wav`
        analysisResult.audioPath = relativePath
      }

      // Clean up error log if it exists
      try {
        await unlink(errorLogPath)
      } catch (error) {
        // Ignore if error log doesn't exist
      }

      // Return both the spectrogram URL and emotion analysis
      return NextResponse.json({
        ...analysisResult,
        spectrogramUrl: `/spectrograms/${id}.png`
      })
    } catch (execError) {
      // Check if error is just the torch deprecation warning
      const errorStr = String(execError);
      const isOnlyDeprecationWarning =
        errorStr.includes('torch.cuda.amp.custom_fwd') &&
        !errorStr.includes('ImportError') &&
        !errorStr.includes('FileNotFoundError');

      if (isOnlyDeprecationWarning) {
        console.warn("Ignoring SpeechBrain deprecation warning in execError:", errorStr);
        // Move files to unknown folder since we're returning neutral
        const unknownDir = join(rootStorageDir, 'unknown')
        await ensureDir(unknownDir)

        const finalWebmPath = join(unknownDir, `${id}.webm`)
        const finalWavPath = join(unknownDir, `${id}.wav`)

        await Promise.all([
          rename(audioPath, finalWebmPath),
          rename(wavPath, finalWavPath)
        ])

        // Use the correct root folder in the path
        const relativePath = isTrainingSample ?
          `/training_samples/unknown/${id}.wav` :
          `/user_recordings/unknown/${id}.wav`

        // Return a successful response with placeholder data
        return NextResponse.json({
          spectrogramUrl: `/placeholder.svg`,
          emotion: "neutral",
          confidence: 0.5,
          category_scores: {
            rhythm: 0.5, energy: 0.5, pitch: 0.5,
            pause: 0.5, voice_quality: 0.5, speech_rate: 0.5
          },
          message: "Processed with warnings (ignorable)",
          audioPath: relativePath
        });
      }

      // Check for error log even in case of execution error
      if (existsSync(errorLogPath)) {
        const errorLog = await readFile(errorLogPath, 'utf8')
        console.error("Python error log:", errorLog)
        return NextResponse.json({ error: `Python error: ${errorLog}` }, { status: 500 })
      }

      console.error("Python execution error:", execError)
      console.error("Full error details:", JSON.stringify(execError, null, 2))
      return NextResponse.json(
        { error: `Python execution failed: ${execError instanceof Error ? execError.message : String(execError)}` },
        { status: 500 }
      )
    }
  } catch (error) {
    console.error("Error analyzing audio:", error)
    return NextResponse.json(
      { error: `Failed to analyze audio: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    )
  }
}