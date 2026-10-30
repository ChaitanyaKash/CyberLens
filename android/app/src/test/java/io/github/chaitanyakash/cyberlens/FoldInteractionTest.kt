package io.github.chaitanyakash.cyberlens

import androidx.compose.material3.adaptive.HingeInfo
import androidx.compose.material3.adaptive.Posture
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], qualifiers = "w824dp-h900dp-mdpi", application = android.app.Application::class)
class FoldInteractionTest {
    @get:Rule val compose = createComposeRule()

    @Test fun settingsSurviveFoldAndUnfoldWithControlsClearOfTheHinge() {
        val posture = mutableStateOf(Posture())
        compose.setContent {
            CompositionLocalProvider(LocalPostureOverride provides posture.value) {
                CyberLensShell(reduceMotion = true, initialDestination = Destination.Profile, initialPage = "settings")
            }
        }
        compose.onNodeWithText("Reduce motion").assertExists()
        compose.runOnIdle {
            posture.value = Posture(hingeList = listOf(HingeInfo(Rect(400f, 0f, 424f, 900f), false, true, true, true)))
        }
        compose.onNodeWithText("Reduce motion").assertExists()
        assertTrue(compose.onNodeWithTag("navigation").fetchSemanticsNode().boundsInRoot.right <= 400f)
        assertTrue(compose.onNodeWithTag("pane_content").fetchSemanticsNode().boundsInRoot.left >= 424f)
        compose.runOnIdle { posture.value = Posture() }
        compose.onNodeWithText("Reduce motion").assertExists()
        compose.onNodeWithText("Back").performClick()
        compose.onNodeWithText("Make CyberLens yours").assertExists()
    }

    @Test fun tabletopKeepsNavigationBelowAndContentAboveTheHinge() {
        compose.setContent {
            CompositionLocalProvider(LocalPostureOverride provides Posture(true,
                listOf(HingeInfo(Rect(0f, 420f, 824f, 440f), false, false, true, true)))) {
                CyberLensShell(reduceMotion = true)
            }
        }
        assertTrue(compose.onNodeWithTag("pane_content").fetchSemanticsNode().boundsInRoot.bottom <= 420f)
        assertTrue(compose.onNodeWithTag("navigation").fetchSemanticsNode().boundsInRoot.top >= 440f)
        compose.onNodeWithText("History").performClick()
        compose.onNodeWithText("A clearer picture, saved here").assertExists()
    }
}
