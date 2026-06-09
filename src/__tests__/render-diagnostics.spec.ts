import { describe, expect, it } from 'vitest'
import { MujocoRenderDiagnosticsCollector } from '../diagnostics/render-diagnostics'

describe('MujocoRenderDiagnosticsCollector', () => {
  it('collects structured warnings and deduplicates explicit ids', () => {
    const collector = new MujocoRenderDiagnosticsCollector()

    collector.add({
      id: 'missing-runtime-field:site_xpos',
      severity: 'warning',
      category: 'missing-runtime-field',
      objectType: 'site',
      message: 'site_xpos missing',
    })
    collector.add({
      id: 'missing-runtime-field:site_xpos',
      severity: 'warning',
      category: 'missing-runtime-field',
      objectType: 'site',
      message: 'site_xpos missing again',
    })
    collector.add({
      severity: 'warning',
      category: 'unsupported-geom',
      objectType: 'geom',
      objectId: 2,
      message: 'SDF unsupported',
    })

    const diagnostics = collector.snapshot()

    expect(diagnostics.warningCount).toBe(2)
    expect(diagnostics.unsupportedCount).toBe(1)
    expect(diagnostics.errorCount).toBe(0)
    expect(diagnostics.warnings[0].message).toBe('site_xpos missing')
  })
})
