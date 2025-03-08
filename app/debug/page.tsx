"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function DebugPage() {
  const [pythonInfo, setPythonInfo] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const checkPythonEnvironment = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/python-check")
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to check Python environment")
      }

      setPythonInfo(data)
    } catch (error) {
      console.error("Error checking Python environment:", error)
      setError(error instanceof Error ? error.message : String(error))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkPythonEnvironment()
  }, [])

  return (
    <div className="container py-12">
      <Card>
        <CardHeader>
          <CardTitle>Python Environment Debug</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p>Loading Python environment information...</p>
          ) : error ? (
            <div className="p-4 mb-4 text-red-700 bg-red-100 rounded-md">
              <p>Error: {error}</p>
            </div>
          ) : pythonInfo ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium">Python Version</h3>
                <pre className="p-3 mt-2 overflow-auto text-sm bg-gray-100 rounded-md">{pythonInfo.python}</pre>
              </div>

              <div>
                <h3 className="text-lg font-medium">Required Packages</h3>
                <pre className="p-3 mt-2 overflow-auto text-sm bg-gray-100 rounded-md whitespace-pre-wrap">
                  {pythonInfo.packages}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-medium">Node Environment</h3>
                <pre className="p-3 mt-2 overflow-auto text-sm bg-gray-100 rounded-md">{pythonInfo.environment}</pre>
              </div>
            </div>
          ) : (
            <p>No information available</p>
          )}

          <Button onClick={checkPythonEnvironment} className="mt-4" disabled={loading}>
            Refresh Information
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

