package io.github.chaitanyakash.cyberlens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.text.TextLayoutResult
import androidx.test.espresso.Espresso
import io.github.chaitanyakash.cyberlens.core.designsystem.CyberLensTheme
import io.github.chaitanyakash.cyberlens.core.designsystem.CLToggleRow
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], application = android.app.Application::class)
class ComponentInteractionTest {
    @get:Rule val compose = createComposeRule()

    private fun samples(group: Int) {
        compose.setContent {
            CyberLensTheme(reduceMotion = true) {
                Surface(Modifier.fillMaxSize()) { CatalogueSamples(group) }
            }
        }
    }

    @Test fun questionChoiceClosesTheSheet() {
        samples(3)
        compose.onNodeWithText("Preview question sheet").performClick()
        compose.onNodeWithText("Did the caller ask for a payment?").assertIsDisplayed()
        compose.onNodeWithText("Yes").performClick()
        compose.onNodeWithText("Did the caller ask for a payment?").assertDoesNotExist()
        compose.onNodeWithText("Preview question sheet").assertIsDisplayed()
    }

    @Test fun questionSheetKeepsTheCallerFontScale() {
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(1f, 2f)) {
                CyberLensTheme(reduceMotion = true) {
                    CatalogueSamples(3, initialSheet = true)
                }
            }
        }
        val layouts = mutableListOf<TextLayoutResult>()
        compose.onNodeWithText("Did the caller ask for a payment?")
            .performSemanticsAction(SemanticsActions.GetTextLayoutResult) { it(layouts) }
        org.junit.Assert.assertEquals(2f, layouts.single().layoutInput.density.fontScale, 0f)
    }

    @Test fun systemBackDismissesOnlyTheQuestionSheet() {
        samples(3)
        compose.onNodeWithText("Preview question sheet").performClick()
        compose.onNodeWithText("Did the caller ask for a payment?").assertIsDisplayed()
        Espresso.pressBack()
        compose.onNodeWithText("Did the caller ask for a payment?").assertDoesNotExist()
        compose.onNodeWithText("Preview question sheet").assertIsDisplayed()
    }

    @Test fun sourceDetailsCanBeOpenedAndClosed() {
        samples(1)
        compose.onNodeWithText("[S1] Official source").performClick()
        compose.onNodeWithText("Official source details will appear here.").assertIsDisplayed()
        compose.onNodeWithText("[S1] Official source").performClick()
        compose.onNodeWithText("Official source details will appear here.").assertDoesNotExist()
    }

    @Test fun switchAnnouncesTheChangedStateAndInheritedSettingIsReadOnly() {
        compose.setContent {
            CyberLensTheme(reduceMotion = true) {
                var checked by remember { mutableStateOf(false) }
                Column {
                    CLToggleRow("Preview switch", checked, { checked = it })
                    CLToggleRow("Inherited switch", true, {}, enabled = false)
                }
            }
        }
        compose.onNodeWithText("Preview switch").assertIsOff().performClick().assertIsOn()
        compose.onAllNodesWithText("Preview switch").assertCountEquals(1)
        compose.onNodeWithText("Inherited switch").assertIsOn().assertIsNotEnabled()
    }
}
