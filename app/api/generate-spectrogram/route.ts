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
    const audioPath = join(tempDir, `${id}.webm`).replace(/\\/g, '/')
    const wavPath = join(tempDir, `${id}.wav`).replace(/\\/g, '/')
    const spectrogramPath = join(tempDir, `${id}_spectrogram.png`).replace(/\\/g, '/')
    const errorLogPath = join(tempDir, `${id}_error.log`).replace(/\\/g, '/')

    console.log("Paths:", {
      tempDir,
      audioPath,
      wavPath,
      spectrogramPath,
      errorLogPath
    })

    // Create public directory for spectrograms if it doesn't exist
    const publicDir = join(process.cwd(), "public")
    const spectrogramsDir = join(publicDir, "spectrograms")
    await ensureDir(spectrogramsDir)

    const publicSpectrogramPath = join(spectrogramsDir, `${id}.png`).replace(/\\/g, '/')

    // Write the audio file to disk
    const audioBuffer = Buffer.from(await audioFile.arrayBuffer())
    console.log("Writing audio file, size:", audioBuffer.length, "bytes")
    await writeFile(audioPath, audioBuffer)
    console.log("Audio file written successfully")

    // Convert WebM to WAV using ffmpeg
    try {
      console.log("Converting WebM to WAV...")
      await execPromise(`ffmpeg -i "${audioPath}" "${wavPath}"`)
      console.log("Conversion successful")
    } catch (error) {
      console.error("FFmpeg conversion failed:", error)
      return NextResponse.json({ error: "Failed to convert audio format" }, { status: 500 })
    }

    // Run the Python script directly with inline code
    const pythonScript = `
import sys
import traceback
import os

# Add the project root directory to Python path
project_root = '${process.cwd().replace(/\\/g, '/')}'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

from audio_processor import AudioProcessor

try:
    # Print debugging info
    audio_path = '${wavPath}'  # Use the WAV file instead of WebM
    output_path = '${publicSpectrogramPath}'
    
    print(f"Audio file exists: {os.path.exists(audio_path)}")
    print(f"Audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'file not found'}")
    
    # Process audio
    processor = AudioProcessor()
    result = processor.process_audio_file(audio_path, generate_spectrogram=True, spectrogram_path=output_path)
    
    # Print results as JSON for parsing
    import json
    print("RESULT_JSON:" + json.dumps({
        "emotion": result.label,
        "confidence": float(result.confidence),
        "features": result.features
    }))

except Exception as e:
    error_msg = f"Error: {str(e)}\\nTraceback:\\n{traceback.format_exc()}"
    print(error_msg)  # This will go to stderr
    with open('${errorLogPath}', 'w') as f:
        f.write(error_msg)
    sys.exit(1)
    `

    // Write the Python script to a temporary file
    const scriptPath = join(tempDir, `${id}_script.py`).replace(/\\/g, '/')
    await writeFile(scriptPath, pythonScript)

    // Execute the Python script using the venv Python path
    const pythonPath = join(process.cwd(), 'venv', 'Scripts', 'python.exe').replace(/\\/g, '/')
    
    try {
      const { stdout, stderr } = await execPromise(`"${pythonPath}" "${scriptPath}"`)
      
      // Clean up temporary files
      try {
        await Promise.all([
          unlink(audioPath),
          unlink(wavPath),
          unlink(scriptPath),
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
        features: analysisResult.features,
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