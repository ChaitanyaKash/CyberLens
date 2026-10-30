package io.github.chaitanyakash.cyberlens.core.designsystem

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.ui.graphics.lerp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.*

object CLTokens {
    val blue = Color(0xFF0A84FF)
    val indigo = Color(0xFF5E5CE6)
    val scam = Color(0xFFCF2821)
    val amber = Color(0xFF995000)
    val green = Color(0xFF187333)
    val cardRadius = 16.dp
    val sheetRadius = 24.dp
    val chipRadius = 10.dp
}
@Composable
fun CyberLensTheme(
    dark: Boolean = isSystemInDarkTheme(),
    reduceMotion: Boolean = false,
    reduceTransparency: Boolean = false,
    content: @Composable () -> Unit,
) {
    val typography = cyberLensTypography()
    val darkPalette = remember { darkColorScheme(
        primary = Color(0xFF86BBFF), onPrimary = Color(0xFF002D57),
        background = Color.Black, onBackground = Color.White, surface = Color(0xFF1C1C1E),
        primaryContainer = Color(0xFF15365B), onPrimaryContainer = Color(0xFFDDEBFF),
        secondary = Color(0xFF86BBFF), secondaryContainer = Color(0xFF15365B), onSecondaryContainer = Color(0xFFDDEBFF),
        surfaceContainer = Color(0xFF1C1C1E), surfaceContainerHigh = Color(0xFF2C2C2E),
        surfaceContainerLow = Color(0xFF1C1C1E), surfaceContainerLowest = Color.Black,
        surfaceContainerHighest = Color(0xFF2C2C2E),
        surfaceVariant = Color(0xFF2C2C2E), onSurface = Color.White,
        onSurfaceVariant = Color(0xFFB8B8BF),
    ) }
    val lightPalette = remember { lightColorScheme(
        primary = Color(0xFF0065CA), onPrimary = Color.White,
        background = Color(0xFFF2F2F7), onBackground = Color.Black, surface = Color.White,
        primaryContainer = Color(0xFFDDEBFF), onPrimaryContainer = Color(0xFF15365B),
        secondary = Color(0xFF0065CA), secondaryContainer = Color(0xFFDDEBFF), onSecondaryContainer = Color(0xFF15365B),
        surfaceContainer = Color.White, surfaceContainerHigh = Color(0xFFF7F7FA),
        surfaceContainerLow = Color(0xFFF7F7FA), surfaceContainerLowest = Color.White,
        surfaceContainerHighest = Color(0xFFF2F2F7),
        surfaceVariant = Color(0xFFF7F7FA), onSurface = Color.Black,
        onSurfaceVariant = Color(0xFF606068),
    ) }
    val appearance by animateFloatAsState(if (dark) 1f else 0f,
        if (reduceMotion) tween(180) else CLMotion.micro, label = "appearance")
    val scheme = blendPalette(lightPalette, darkPalette, appearance)
    CompositionLocalProvider(LocalReduceMotion provides reduceMotion,
        LocalReduceTransparency provides reduceTransparency) {
        MaterialTheme(colorScheme = scheme, typography = typography, content = content)
    }
}


private fun blendPalette(light: ColorScheme, dark: ColorScheme, fraction: Float): ColorScheme {
    if (fraction <= 0f) return light
    if (fraction >= 1f) return dark
    return light.copy(
        primary = lerp(light.primary, dark.primary, fraction),
        onPrimary = lerp(light.onPrimary, dark.onPrimary, fraction),
        primaryContainer = lerp(light.primaryContainer, dark.primaryContainer, fraction),
        onPrimaryContainer = lerp(light.onPrimaryContainer, dark.onPrimaryContainer, fraction),
        inversePrimary = lerp(light.inversePrimary, dark.inversePrimary, fraction),
        secondary = lerp(light.secondary, dark.secondary, fraction),
        onSecondary = lerp(light.onSecondary, dark.onSecondary, fraction),
        secondaryContainer = lerp(light.secondaryContainer, dark.secondaryContainer, fraction),
        onSecondaryContainer = lerp(light.onSecondaryContainer, dark.onSecondaryContainer, fraction),
        tertiary = lerp(light.tertiary, dark.tertiary, fraction),
        onTertiary = lerp(light.onTertiary, dark.onTertiary, fraction),
        tertiaryContainer = lerp(light.tertiaryContainer, dark.tertiaryContainer, fraction),
        onTertiaryContainer = lerp(light.onTertiaryContainer, dark.onTertiaryContainer, fraction),
        background = lerp(light.background, dark.background, fraction),
        onBackground = lerp(light.onBackground, dark.onBackground, fraction),
        surface = lerp(light.surface, dark.surface, fraction),
        onSurface = lerp(light.onSurface, dark.onSurface, fraction),
        surfaceVariant = lerp(light.surfaceVariant, dark.surfaceVariant, fraction),
        onSurfaceVariant = lerp(light.onSurfaceVariant, dark.onSurfaceVariant, fraction),
        surfaceTint = lerp(light.surfaceTint, dark.surfaceTint, fraction),
        inverseSurface = lerp(light.inverseSurface, dark.inverseSurface, fraction),
        inverseOnSurface = lerp(light.inverseOnSurface, dark.inverseOnSurface, fraction),
        error = lerp(light.error, dark.error, fraction),
        onError = lerp(light.onError, dark.onError, fraction),
        errorContainer = lerp(light.errorContainer, dark.errorContainer, fraction),
        onErrorContainer = lerp(light.onErrorContainer, dark.onErrorContainer, fraction),
        outline = lerp(light.outline, dark.outline, fraction),
        outlineVariant = lerp(light.outlineVariant, dark.outlineVariant, fraction),
        scrim = lerp(light.scrim, dark.scrim, fraction),
        surfaceBright = lerp(light.surfaceBright, dark.surfaceBright, fraction),
        surfaceDim = lerp(light.surfaceDim, dark.surfaceDim, fraction),
        surfaceContainer = lerp(light.surfaceContainer, dark.surfaceContainer, fraction),
        surfaceContainerHigh = lerp(light.surfaceContainerHigh, dark.surfaceContainerHigh, fraction),
        surfaceContainerHighest = lerp(light.surfaceContainerHighest, dark.surfaceContainerHighest, fraction),
        surfaceContainerLow = lerp(light.surfaceContainerLow, dark.surfaceContainerLow, fraction),
        surfaceContainerLowest = lerp(light.surfaceContainerLowest, dark.surfaceContainerLowest, fraction),
    )
}
