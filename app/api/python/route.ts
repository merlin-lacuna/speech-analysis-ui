import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { join } from "path"

export async function POST(request: NextRequest) {
  try {
    const { command, args = [] } = await request.json()

    if (!command) {
      return NextResponse.json({ error: "No command provided" }, { status: 400 })
    }

    const scriptPath = join(process.cwd(), "scripts", "speech_feature_extractor.py")

    return new Promise((resolve) => {
      // Run the Python script with the command and args
      const process = spawn("python", [scriptPath, command, ...args])

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
          console.error("Python script error:", stderr)
          resolve(
            NextResponse.json(
              {
                error: stderr || "Python script execution failed",
                logs: stderr
              },
              { status: 500 }
            )
          )
        } else {
          try {
            // Try to parse the output as JSON
            const result = JSON.parse(stdout)
            resolve(NextResponse.json(result))
          } catch {
            // If not JSON, return as-is
            resolve(NextResponse.json({ result: stdout }))
          }
        }
      })
    })
  } catch (error) {
    console.error("Error executing Python script:", error)
    return NextResponse.json(
      {
        error: "Failed to execute Python script",
        logs: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    )
  }
}

