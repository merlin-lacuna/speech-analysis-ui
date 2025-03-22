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
    console.log("Received sample ingestion request")
    const formData = await request.formData()
    const audioFile = formData.get("audio") as File
    const label = formData.get("label") as string

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
    const audioPath = join(tempDir, `${id}.webm`)
    const wavPath = join(tempDir, `${id}.wav`)
    const errorLogPath = join(tempDir, `${id}_error.log`)

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
      const ffmpegPath = process.platform === 'win32' 
        ? 'ffmpeg' // On Windows, assumes ffmpeg is in PATH
        : join(process.cwd(), 'bin', 'ffmpeg') // On Linux, use local binary
      
      await execPromise(`"${ffmpegPath}" -i "${audioPath}" "${wavPath}"`)
      console.log("Conversion successful")
    } catch (error) {
      console.error("FFmpeg conversion failed:", error)
      return NextResponse.json({ error: "Failed to convert audio format" }, { status: 500 })
    }

    // Get path to Python script and database
    const scriptPath = join(process.cwd(), 'scripts', 'api', 'ingest_sample.py')
    const databasePath = join(process.cwd(), 'emotion_database')
    
    // Execute the Python script using the venv Python path (platform-specific)
    const pythonPath = process.platform === 'win32' 
      ? join(process.cwd(), 'venv', 'Scripts', 'python.exe')
      : join(process.cwd(), 'venv', 'bin', 'python')
    
    try {
      // Run the dedicated Python script with parameters
      const { stdout, stderr } = await execPromise(
        `"${pythonPath}" "${scriptPath}" "${wavPath}" "${label}" "${databasePath}" "${errorLogPath}"`
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
        // Ignore known deprecation warnings about torch.cuda.amp.custom_fwd
        if (stderr.includes('torch.cuda.amp.custom_fwd') && 
            !stderr.includes('ImportError') && 
            !stderr.includes('FileNotFoundError') &&
            stderr.includes('INFO:root:Successfully added and saved sample')) {
          console.warn("Ignoring SpeechBrain deprecation warning in stderr:", stderr)
        } else {
          console.error("Python script stderr:", stderr)
          return NextResponse.json({ error: `Python error: ${stderr}` }, { status: 500 })
        }
      }

      console.log("Python script output:", stdout)

      // Parse the result from stdout
      const resultMatch = stdout.match(/RESULT_JSON:(.+)/)
      if (resultMatch) {
        const result = JSON.parse(resultMatch[1])
        return NextResponse.json(result)
      }

      // Return success response if no JSON result was found
      return NextResponse.json({
        success: true,
        message: "Sample successfully ingested",
      })
    } catch (execError) {
      // Check if error is just the torch deprecation warning
      const errorStr = String(execError);
      const isOnlyDeprecationWarning = 
        errorStr.includes('torch.cuda.amp.custom_fwd') && 
        !errorStr.includes('ImportError') && 
        !errorStr.includes('FileNotFoundError') &&
        (errorStr.includes('Successfully added and saved sample') ||
         errorStr.includes('INFO:root:Successfully processed'));

      if (isOnlyDeprecationWarning) {
        console.warn("Ignoring SpeechBrain deprecation warning in execError:", errorStr);
        // Return a successful response
        return NextResponse.json({
          success: true,
          message: "Sample successfully ingested (with ignorable warnings)",
        });
      }
      
      // Check for error log even in case of execution error
      if (existsSync(errorLogPath)) {
        const errorLog = await readFile(errorLogPath, 'utf8');
        
        // Check if error log only contains deprecation warnings but also has success messages
        if (errorLog.includes('torch.cuda.amp.custom_fwd') && 
            (errorLog.includes('Successfully added and saved sample') ||
             errorLog.includes('INFO:root:Successfully processed')) &&
            !errorLog.includes('ImportError') && 
            !errorLog.includes('FileNotFoundError')) {
          console.warn("Ignoring SpeechBrain deprecation warning in error log:", errorLog);
          // Return a successful response
          return NextResponse.json({
            success: true,
            message: "Sample successfully ingested (with ignorable warnings)",
          });
        }
        
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
    console.error("Error ingesting sample:", error)
    return NextResponse.json(
      { error: `Failed to ingest sample: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    )
  }
}