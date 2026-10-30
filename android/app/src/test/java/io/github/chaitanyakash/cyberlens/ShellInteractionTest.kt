@file:OptIn(androidx.compose.ui.test.ExperimentalTestApi::class)

package io.github.chaitanyakash.cyberlens

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.junit4.StateRestorationTester
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalInputModeManager
import androidx.compose.ui.input.InputMode
import androidx.compose.ui.input.InputModeManager
import androidx.test.platform.app.InstrumentationRegistry
import androidx.compose.ui.unit.Density
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.Role
import org.robolectric.RuntimeEnvironment
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], qualifiers = "w384dp-h832dp-mdpi", application = android.app.Application::class)
class ShellInteractionTest {
    @get:Rule val compose = createComposeRule()

    @Test fun navigationAndSettingsRemainUsable() {
        compose.setContent { CyberLensShell(reduceMotion = true) }
        compose.onNodeWithText("Messages").performClick()
        compose.onNodeWithText("Your messages, with context").assertExists()
        compose.onNodeWithText("History").performClick()
        compose.onNodeWithText("A clearer picture, saved here").assertExists()
        compose.onNodeWithText("Profile").performClick()
        compose.onNodeWithText("Settings").performClick()
        compose.onNodeWithText("Reduce motion").assertExists()
        compose.onNodeWithText("Back").performClick()
        compose.onNodeWithText("Make CyberLens yours").assertExists()
    }

    @Test fun emergencyButtonUsesOnlyAnExplicitCallback() {
        var dialled = false
        compose.setContent { CyberLensShell(reduceMotion = true, onEmergency = { dialled = true }) }
        repeat(2) { compose.onNode(hasScrollAction()).performTouchInput { swipeUp(durationMillis = 800) } }
        val button = compose.onNodeWithText("Call 1930")
        val navigation = compose.onNodeWithTag("navigation").fetchSemanticsNode().boundsInRoot
        org.junit.Assert.assertTrue(button.fetchSemanticsNode().boundsInRoot.bottom <= navigation.top + 1f)
        button.performClick()
        org.junit.Assert.assertTrue(dialled)
    }

    @Test fun currentScreenSurvivesSavedStateRestoration() {
        val restoration = StateRestorationTester(compose)
        restoration.setContent { CyberLensShell(reduceMotion = true) }
        compose.onNodeWithText("Profile").performClick()
        compose.onNodeWithText("Settings").performClick()
        restoration.emulateSavedInstanceStateRestore()
        compose.onNodeWithText("Reduce motion").assertExists()
        compose.onNodeWithText("Back").performClick()
        compose.onNodeWithText("Make CyberLens yours").assertExists()
    }

    @Test @Config(qualifiers = "w320dp-h640dp-mdpi")
    fun allFiveTabsRemainVisibleAtTwoHundredPercent() {
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(1f, 2f)) {
                CyberLensShell(reduceMotion = true)
            }
        }
        val navigation = compose.onNodeWithTag("navigation").fetchSemanticsNode().boundsInRoot
        org.junit.Assert.assertEquals(320f, navigation.width, 1f)
        val tabs = compose.onAllNodes(
            SemanticsMatcher.expectValue(SemanticsProperties.Role, Role.Tab) and
                hasAnyAncestor(hasTestTag("navigation")))
        tabs.assertCountEquals(5)
        tabs.fetchSemanticsNodes().forEach { node ->
            val bounds = node.boundsInRoot
            org.junit.Assert.assertTrue(bounds.width >= 48f && bounds.height >= 48f)
            org.junit.Assert.assertTrue(bounds.left >= navigation.left && bounds.right <= navigation.right + 1f)
            org.junit.Assert.assertTrue(bounds.top >= navigation.top && bounds.bottom <= navigation.bottom + 1f)
        }
        val layouts = mutableListOf<androidx.compose.ui.text.TextLayoutResult>()
        compose.onNodeWithTag("navigation_label").performSemanticsAction(SemanticsActions.GetTextLayoutResult) { it(layouts) }
        org.junit.Assert.assertEquals(1, layouts.single().lineCount)
        org.junit.Assert.assertEquals(2f, layouts.single().layoutInput.density.fontScale, 0f)
        compose.onNodeWithContentDescription("Messages").performClick()
        compose.onNodeWithTag("navigation_label").assertTextEquals("Messages")
    }

    @Test @Config(qualifiers = "w320dp-h640dp-mdpi")
    fun keyboardFocusStaysAboveTheNavigationOverlay() {
        InstrumentationRegistry.getInstrumentation().setInTouchMode(false)
        lateinit var inputMode: InputModeManager
        compose.setContent {
            val currentInputMode = LocalInputModeManager.current
            SideEffect { inputMode = currentInputMode }
            CompositionLocalProvider(LocalDensity provides Density(1f, 2f)) {
                CyberLensShell(reduceMotion = true, initialDestination = Destination.Profile)
            }
        }
        compose.runOnIdle { org.junit.Assert.assertTrue(inputMode.requestInputMode(InputMode.Keyboard)) }
        compose.onRoot().performKeyInput { pressKey(Key.Tab) }
        val settings = compose.onNodeWithText("Settings")
        settings.performSemanticsAction(SemanticsActions.RequestFocus) { it() }
        settings.assertIsFocused()
        val navigation = compose.onNodeWithTag("navigation").fetchSemanticsNode().boundsInRoot
        org.junit.Assert.assertTrue(settings.fetchSemanticsNode().boundsInRoot.bottom <= navigation.top + 1f)
    }
}
