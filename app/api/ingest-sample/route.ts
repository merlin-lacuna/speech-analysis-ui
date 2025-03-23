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
    console.log("Received sample ingestion request")
    const formData = await request.formData()
    const audioFile = formData.get("audio") as File
    const label = formData.get("label") as string
    const name = formData.get("name") as string || "Unknown"

    if (!audioFile || !label) {
      console.error("Missing audio file or label")
      return NextResponse.json({ error: "Missing audio file or label" }, { status: 400 })
    }

    console.log("Sample received:", {
      name: audioFile.name,
      type: audioFile.type,
      size: audioFile.size,
      label: label
    })

    // Create unique filenames using OS temp directory
    const id = randomUUID()
    const tempDir = os.tmpdir()
    const audioPath = join(tempDir, `${id}.webm`).replace(/\\/g, '/')
    const wavPath = join(tempDir, `${id}.wav`).replace(/\\/g, '/')
    const errorLogPath = join(tempDir, `${id}_error.log`).replace(/\\/g, '/')

    // Create training directory if it doesn't exist
    const trainingDir = join(process.cwd(), "audio_data", "training")
    await ensureDir(trainingDir)

    console.log("Paths:", {
      tempDir,
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
      await execPromise(`ffmpeg -i "${audioPath}" "${wavPath}"`)
      console.log("Conversion successful")
    } catch (error) {
      console.error("FFmpeg conversion failed:", error)
      return NextResponse.json({ error: "Failed to convert audio format" }, { status: 500 })
    }

    // Save the WAV file with the label and name
    const finalAudioPath = join(trainingDir, `${label}_${name}_${id}.wav`).replace(/\\/g, '/')
    await rename(wavPath, finalAudioPath)

    // Run the Python script to ingest the sample
    const pythonScript = `
import sys
import traceback
import os

# Add the scripts directory to Python path
scripts_dir = os.path.join('${process.cwd().replace(/\\/g, '/')}', 'scripts')
sys.path.append(scripts_dir)

from speech_feature_extractor import HybridFeatureExtractor

try:
    # Print debugging info
    audio_path = '${finalAudioPath}'  # Use the final WAV file path
    label = '${label}'
    database_path = os.path.join('${process.cwd().replace(/\\/g, '/')}', 'emotion_database')  # Path to our emotion database
    
    print(f"Scripts directory: {scripts_dir}")
    print(f"Database path: {database_path}")
    print(f"Audio file exists: {os.path.exists(audio_path)}")
    print(f"Audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'file not found'}")
    
    # Initialize feature extractor
    extractor = HybridFeatureExtractor()
    
    # Load existing database if it exists
    if os.path.exists(f"{database_path}_index.faiss"):
        print("Loading existing database...")
        extractor.load_database(database_path)
    
    # Add the new sample
    print(f"Adding sample with label: {label}")
    extractor.add_sample(audio_path, label)
    
    # Save the updated database
    print("Saving database...")
    extractor.save_database(database_path)
    print("Sample successfully ingested")

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
    const pythonPath = join(process.cwd(), 'venv', process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python').replace(/\\/g, '/')

    try {
      const { stdout, stderr } = await execPromise(`"${pythonPath}" "${scriptPath}"`)

      // Clean up temporary files
      try {
        await Promise.all([
          unlink(audioPath),
          unlink(scriptPath),
          unlink(errorLogPath).catch(() => { })  // Error log might not exist
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

      // Return success response
      return NextResponse.json({
        success: true,
        message: "Sample successfully ingested",
      })
    } catch (error) {
      console.error("Python execution error:", error)
      return NextResponse.json(
        { error: `Python execution failed: ${error instanceof Error ? error.message : String(error)}` },
        { status: 500 }
      )
    }
  } catch (error) {
    console.error("Error ingesting sample:", error)
    return NextResponse.json(
      { error: `Failed to ingest sample: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    )
  }
} 