package io.github.chaitanyakash.cyberlens.core.designsystem

import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.font.*
import androidx.compose.ui.unit.*

private val Inter = FontFamily(
    Font(R.font.inter, FontWeight.Normal),
    Font(R.font.inter, FontWeight.Medium),
    Font(R.font.inter, FontWeight.SemiBold),
    Font(R.font.inter, FontWeight.Bold),
)
private val Devanagari = FontFamily(
    Font(R.font.noto_sans_devanagari, FontWeight.Normal),
    Font(R.font.noto_sans_devanagari, FontWeight.SemiBold),
    Font(R.font.noto_sans_devanagari, FontWeight.Bold),
)
val CLWordmarkFont = FontFamily(Font(R.font.sora, FontWeight.SemiBold))

@OptIn(ExperimentalTextApi::class)
@Composable
internal fun cyberLensTypography(): Typography {
    val hindi = LocalConfiguration.current.locales[0].language == "hi"
    val font = if (hindi) Devanagari else Inter
    fun type(size: Int, leading: Float, weight: FontWeight = FontWeight.Normal, tracking: Float = 0f) =
        TextStyle(fontFamily = if (hindi) font else FontFamily(Font(R.font.inter, weight,
            variationSettings = FontVariation.Settings(FontVariation.Setting("opsz", size.toFloat()),
                FontVariation.weight(weight.weight)))), fontSize = size.sp,
            lineHeight = (size * (leading + if (hindi) 0.1f else 0f)).sp,
            fontWeight = weight, letterSpacing = tracking.em)
    return Typography(
        displayLarge = type(34, 1.1f, FontWeight.Bold, -0.02f),
        headlineLarge = type(28, 1.2f, FontWeight.Bold, -0.01f),
        headlineMedium = type(22, 1.2f, FontWeight.SemiBold, -0.01f),
        headlineSmall = type(20, 1.2f, FontWeight.SemiBold),
        titleLarge = type(17, 1.4f, FontWeight.SemiBold),
        titleMedium = type(16, 1.4f, FontWeight.SemiBold),
        bodyLarge = type(17, 1.45f),
        bodyMedium = type(16, 1.45f),
        bodySmall = type(15, 1.4f),
        labelLarge = type(17, 1.4f, FontWeight.SemiBold),
        labelMedium = type(13, 1.4f, FontWeight.Medium, 0.01f),
        labelSmall = type(12, 1.4f, FontWeight.Medium, 0.01f),
    )
}
