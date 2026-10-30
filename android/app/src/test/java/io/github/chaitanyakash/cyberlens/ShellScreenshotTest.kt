package io.github.chaitanyakash.cyberlens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.github.takahirom.roborazzi.captureScreenRoboImage
import com.github.takahirom.roborazzi.ExperimentalRoborazziApi
import io.github.chaitanyakash.cyberlens.core.designsystem.*
import io.github.chaitanyakash.cyberlens.core.testing.ScreenMatrix
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.ParameterizedRobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(ParameterizedRobolectricTestRunner::class)
@Config(sdk = [35], application = android.app.Application::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class ShellScreenshotTest(
    private val name: String,
    private val width: Int,
    private val height: Int,
    private val dark: Boolean,
    private val fontScale: Float,
    private val screen: String,
    private val language: String,
) {
    init {
        RuntimeEnvironment.setQualifiers("${language}-r${if (language == "hi") "IN" else "US"}-w${width}dp-h${height}dp-${if (dark) "night" else "notnight"}-mdpi")
    }
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()

    @OptIn(ExperimentalRoborazziApi::class)
    @Test fun screenAtWindowSize() {
        compose.activity.enableEdgeToEdge()
        var renderedLanguage: String? = null
        compose.setContent {
            val currentLanguage = LocalConfiguration.current.locales[0].language
            SideEffect { renderedLanguage = currentLanguage }
            CompositionLocalProvider(LocalDensity provides Density(1f, fontScale)) {
                if (screen.startsWith("sheet_")) {
                    CyberLensShell(dark = dark, reduceMotion = screen.endsWith("reduced"),
                        reduceTransparency = screen.endsWith("opaque"),
                        storage = StorageState.Ready, initialPage = "components",
                        initialCatalogueSection = 3, initialCatalogueSheet = true)
                } else if (screen == "glass") {
                    CyberLensTheme(dark = dark) {
                        Surface(Modifier.fillMaxSize()) {
                            Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) { GlassSample() }
                        }
                    }
                } else if (screen.startsWith("sample_")) {
                    val group = screen.split('_')[1].toInt()
                    CyberLensTheme(dark = dark, reduceMotion = screen.endsWith("reduced"),
                        reduceTransparency = screen.endsWith("opaque")) {
                        Surface(Modifier.fillMaxSize()) {
                            Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) {
                                CatalogueSamples(group)
                            }
                        }
                    }
                } else {
                    CyberLensShell(dark = dark, storage = StorageState.Ready,
                        initialDestination = Destination.entries.firstOrNull { it.name.lowercase() == screen }
                            ?: Destination.Protection,
                        initialPage = if (screen in listOf("settings", "components")) screen else "")
                }
            }
        }
        if (screen.startsWith("sheet_")) compose.onNode(isDialog()).assertIsDisplayed()
        else compose.onRoot().assertIsDisplayed()
        org.junit.Assert.assertEquals(language, renderedLanguage)
        // Keep the rendered evidence even when a following layout assertion fails.
        captureScreenRoboImage("src/test/screenshots/${screen}_${name}_${language}_${dark}_${fontScale}.png")
        val labels = compose.onAllNodes(SemanticsMatcher.keyIsDefined(SemanticsActions.GetTextLayoutResult), useUnmergedTree = true)
        val clipped = mutableListOf<String>()
        var reconstructedMeasurements = 0
        var largestRounding = 0f
        repeat(labels.fetchSemanticsNodes().size) { index ->
            val layouts = mutableListOf<TextLayoutResult>()
            labels[index].performSemanticsAction(SemanticsActions.GetTextLayoutResult) { it(layouts) }
            layouts.forEach { layout ->
                val visibleLineWidth = (0 until layout.lineCount).maxOfOrNull {
                    layout.getLineRight(it) - layout.getLineLeft(it)
                } ?: 0f
                val widthOverflow = visibleLineWidth - layout.size.width
                val heightOverflow = layout.multiParagraph.height - layout.size.height
                // Compose semantics rebuilds a paragraph at the parent constraint width,
                // while retaining the wrapped text node's size. Compare line advances,
                // not the unused paragraph space. Allow only integer pixel rounding.
                val ellipsized = (0 until layout.lineCount).any { layout.isLineEllipsized(it) }
                if (layout.multiParagraph.didExceedMaxLines || ellipsized || widthOverflow > 1f || heightOverflow > 1f) {
                    clipped += "${layout.layoutInput.text.text}; size=${layout.size}, lineWidth=$visibleLineWidth, paragraphHeight=${layout.multiParagraph.height}"
                } else if (layout.hasVisualOverflow) {
                    reconstructedMeasurements++
                    largestRounding = maxOf(largestRounding, widthOverflow, heightOverflow)
                }
            }
        }
        println("Layout $screen/$name/$language/$fontScale: reconstructed paragraphs=$reconstructedMeasurements, largest rounding=$largestRounding px")
        org.junit.Assert.assertTrue("Clipped text on $screen/$name/$language/$fontScale: ${clipped.joinToString("; ")}", clipped.isEmpty())
    }

    companion object {
        @JvmStatic @ParameterizedRobolectricTestRunner.Parameters(name = "{5}_{0}_{6}_dark={3}_font={4}")
        fun sizes(): List<Array<Any>> {
            val screens = listOf("protection", "messages", "check", "history", "profile", "settings", "components") +
                (0..3).flatMap { listOf("sample_${it}_motion", "sample_${it}_reduced") } +
                listOf("glass", "sheet_motion", "sheet_reduced", "sample_2_opaque", "sheet_opaque")
            val english = ScreenMatrix.windows.flatMap { window ->
                listOf(false, true).flatMap { dark ->
                    (if (window.name == "phone") listOf(1f, 1.3f, 2f) else listOf(1f, 2f)).flatMap { font ->
                        screens.map { screen -> arrayOf<Any>(window.name, window.width, window.height, dark, font, screen, "en") }
                    }
                }
            }
            val phone = ScreenMatrix.windows.first { it.name == "phone" }
            val hindiScreens = screens.filter { it in listOf("protection", "messages", "check", "history", "profile", "settings", "components", "glass", "sheet_reduced", "sample_2_opaque", "sheet_opaque") ||
                (it.startsWith("sample_") && it.endsWith("_motion")) }
            val hindi = listOf(false, true).flatMap { dark -> hindiScreens.map { screen ->
                arrayOf<Any>(phone.name, phone.width, phone.height, dark, 2f, screen, "hi")
            } }
            val window = System.getProperty("cyberlens.screenshot.window", "").orEmpty()
            require(window.isEmpty() || ScreenMatrix.windows.any { it.name == window }) { "Unknown screenshot window: $window" }
            val cases = (english + hindi).filter { window.isEmpty() || it[0] == window }
            // Explicit subsets avoid depending on the runner's display-name formatting.
            return if (java.lang.Boolean.getBoolean("cyberlens.screenshot.smoke")) {
                cases.filter { it[0] == "phone" && it[3] == false && it[4] == 1f && it[5] == "protection" && it[6] == "en" }
            } else cases
        }
    }
}
