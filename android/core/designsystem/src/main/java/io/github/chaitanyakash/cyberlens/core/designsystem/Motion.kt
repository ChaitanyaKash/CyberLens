package io.github.chaitanyakash.cyberlens.core.designsystem

import androidx.compose.animation.core.spring
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.MutableState

object CLMotion {
    val default = spring<Float>(dampingRatio = 1f, stiffness = 322f)
    val sheet = spring<Float>(dampingRatio = 1f, stiffness = 439f)
    val flickedSheet = spring<Float>(dampingRatio = 0.8f, stiffness = 439f)
    val momentum = spring<Float>(dampingRatio = 0.8f, stiffness = 247f)
    val micro = spring<Float>(dampingRatio = 1f, stiffness = 632f)
    fun project(velocity: Float): Float = velocity / 1000f * 0.998f / (1f - 0.998f)
    fun rubberband(overshoot: Float, dimension: Float): Float =
        overshoot * dimension * 0.55f / (dimension + 0.55f * kotlin.math.abs(overshoot))
}
val LocalReduceMotion = staticCompositionLocalOf { false }
val LocalReduceTransparency = staticCompositionLocalOf { false }
val LocalModalProgress = staticCompositionLocalOf<MutableState<Float>?> { null }
