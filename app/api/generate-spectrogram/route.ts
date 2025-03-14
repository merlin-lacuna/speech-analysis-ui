import { type NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir, readFile, unlink } from "fs/promises"
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

    if (!audioFile) {
      console.error("No audio file provided")
      return NextResponse.json({ error: "No audio file provided" }, { status: 400 })
    }

    console.log("Audio file received:", {
      name: audioFile.name,
      type: audioFile.type,
      size: audioFile.size
    })

    // Create unique filenames using OS temp directory
    const id = randomUUID()
    const tempDir = os.tmpdir()
    const audioPath = join(tempDir, `${id}.webm`)
    const wavPath = join(tempDir, `${id}.wav`)
    const errorLogPath = join(tempDir, `${id}_error.log`)

    console.log("Paths:", {
      tempDir,
      audioPath,
      wavPath,
      errorLogPath
    })

    // Create public directory for spectrograms if it doesn't exist
    const publicDir = join(process.cwd(), "public")
    const spectrogramsDir = join(publicDir, "spectrograms")
    await ensureDir(spectrogramsDir)

    const publicSpectrogramPath = join(spectrogramsDir, `${id}.png`)

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
      
      // Clean up temporary files
      try {
        await Promise.all([
          unlink(audioPath),
          unlink(wavPath),
          unlink(errorLogPath).catch(() => {})  // Error log might not exist
        ])
      } catch (error) {
        console.warn("Failed to clean up some temporary files:", error)
      }
      
      // Check for error log
      if (existsSync(errorLogPath)) {
        const errorLog = await readFile(errorLogPath, 'utf8')
        console.error("Python error log:", errorLog)
        return NextResponse.json({ error: `Python error: ${errorLog}` }, { status: 500 })
      }

      if (stderr) {
        console.error("Python script stderr:", stderr)
        return NextResponse.json({ error: `Python error: ${stderr}` }, { status: 500 })
      }

      console.log("Python script output:", stdout)

      // Parse the emotion analysis results from stdout
      const resultMatch = stdout.match(/RESULT_JSON:(.+)/)
      if (!resultMatch) {
        return NextResponse.json({ error: "Failed to parse emotion analysis results" }, { status: 500 })
      }

      const analysisResult = JSON.parse(resultMatch[1])

      // Return both the spectrogram URL and emotion analysis
      return NextResponse.json({
        spectrogramUrl: `/spectrograms/${id}.png`,
        emotion: analysisResult.emotion,
        confidence: analysisResult.confidence,
        category_scores: analysisResult.category_scores,
        message: stdout,
      })
    } catch (execError) {
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