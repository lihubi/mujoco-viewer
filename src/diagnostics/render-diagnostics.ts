export type MujocoRenderWarningSeverity = 'info' | 'warning' | 'error'

export type MujocoRenderWarningCategory =
  | 'unsupported-geom'
  | 'unsupported-deformable'
  | 'unsupported-site'
  | 'unsupported-camera'
  | 'unsupported-light'
  | 'unsupported-tendon'
  | 'unsupported-actuator'
  | 'missing-runtime-field'
  | 'asset-resolution'
  | 'fallback-rendering'

export interface MujocoRenderWarning {
  id: string
  severity: MujocoRenderWarningSeverity
  category: MujocoRenderWarningCategory
  objectType: string
  objectId?: number
  objectName?: string
  message: string
}

export interface MujocoRenderDiagnostics {
  warnings: MujocoRenderWarning[]
  warningCount: number
  unsupportedCount: number
  errorCount: number
}

export class MujocoRenderDiagnosticsCollector {
  private readonly warnings: MujocoRenderWarning[] = []
  private sequence = 0

  clear(): void {
    this.warnings.length = 0
    this.sequence = 0
  }

  add(warning: Omit<MujocoRenderWarning, 'id'> & { id?: string }): MujocoRenderWarning {
    const id = warning.id ?? `${warning.category}:${warning.objectType}:${warning.objectId ?? 'unknown'}:${this.sequence}`
    const existing = this.warnings.find((entry) => entry.id === id)
    if (existing) {
      return existing
    }
    const nextWarning: MujocoRenderWarning = {
      ...warning,
      id,
    }
    this.sequence += 1
    this.warnings.push(nextWarning)
    return nextWarning
  }

  snapshot(): MujocoRenderDiagnostics {
    const warnings = this.warnings.map((warning) => ({ ...warning }))
    return {
      warnings,
      warningCount: warnings.length,
      unsupportedCount: warnings.filter((warning) => warning.category.startsWith('unsupported-')).length,
      errorCount: warnings.filter((warning) => warning.severity === 'error').length,
    }
  }
}

export const createEmptyMujocoRenderDiagnostics = (): MujocoRenderDiagnostics => ({
  warnings: [],
  warningCount: 0,
  unsupportedCount: 0,
  errorCount: 0,
})
