import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { join } from "path"

export async function POST(request: NextRequest) {
  try {
    const { script, args = [] } = await request.json()

    if (!script) {
      return NextResponse.json({ error: "No script provided" }, { status: 400 })
    }

    const scriptPath = join(process.cwd(), script)
    
    // Platform-agnostic Python command
    const pythonCommand = process.platform === 'win32' ? 'python' : 'python3'

    return new Promise((resolve) => {
      const process = spawn(pythonCommand, [scriptPath, ...args])

      let stdout = ""
      let stderr = ""

      process.stdout.on("data", (data) => {
        stdout += data.toString()
      })

      process.stderr.on("data", (data) => {
        stderr += data.toString()
      })

      process.on("close", (code) => {
        if (code !== 0) {
          resolve(NextResponse.json({ error: stderr || "Python script execution failed" }, { status: 500 }))
        } else {
          resolve(NextResponse.json({ result: stdout }))
        }
      })
    })
  } catch (error) {
    console.error("Error executing Python script:", error)
    return NextResponse.json({ error: "Failed to execute Python script" }, { status: 500 })
  }
}

