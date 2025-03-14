import { NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import { join } from "path"

const execPromise = promisify(exec)

export async function GET() {
  try {
    // Platform-agnostic Python command (on Windows it's usually 'python', on Linux/macOS it's often 'python3')
    const pythonCommand = process.platform === 'win32' ? 'python' : 'python3'
    
    // Check if Python is installed and which version
    const { stdout: pythonVersion, stderr: pythonError } = await execPromise(`${pythonCommand} --version`)

    // Check if required packages are installed
    const checkPackages = `
import sys
try:
    import numpy
    import matplotlib
    import scipy
    print("All required packages are installed")
    print(f"numpy: {numpy.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"scipy: {scipy.__version__}")
except ImportError as e:
    print(f"Missing package: {e}")
    sys.exit(1)
    `

    const { stdout: packagesOutput, stderr: packagesError } = await execPromise(`${pythonCommand} -c "${checkPackages}"`)

    return NextResponse.json({
      python: pythonError ? `Error: ${pythonError}` : pythonVersion.trim(),
      packages: packagesError ? `Error: ${packagesError}` : packagesOutput.trim(),
      environment: process.env.NODE_ENV,
    })
  } catch (error) {
    console.error("Error checking Python environment:", error)
    return NextResponse.json(
      { error: `Failed to check Python environment: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 },
    )
  }
}

