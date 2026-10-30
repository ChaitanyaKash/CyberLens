package io.github.chaitanyakash.cyberlens

import androidx.compose.material3.adaptive.HingeInfo
import androidx.compose.material3.adaptive.Posture
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import org.junit.Assert.*
import org.junit.Test

class HingeRegionsTest {
    @Test fun dualScreenKeepsBothPanesClearOfTheOccludedStrip() {
        val hinge = HingeInfo(Rect(400f, 0f, 424f, 900f), false, true, true, true)
        val regions = hingeRegions(824f, 900f, Offset.Zero, Posture(hingeList = listOf(hinge)))!!
        assertEquals(Rect(424f, 0f, 824f, 900f), regions.content)
        assertEquals(Rect(0f, 0f, 400f, 900f), regions.navigation)
        assertEquals(regions.content, regions.sheet)
    }

    @Test fun tabletopPlacesControlsAndQuestionSheetBelowTheCrease() {
        val hinge = HingeInfo(Rect(0f, 420f, 800f, 420f), false, false, true, false)
        val regions = hingeRegions(800f, 900f, Offset.Zero, Posture(true, listOf(hinge)))!!
        assertEquals(420f, regions.content.bottom)
        assertEquals(420f, regions.navigation.top)
        assertEquals(regions.navigation, regions.sheet)
    }

    @Test fun windowCoordinatesAreTranslatedBeforeSplitting() {
        val hinge = HingeInfo(Rect(420f, 30f, 440f, 930f), false, true, true, true)
        val regions = hingeRegions(820f, 900f, Offset(20f, 30f), Posture(hingeList = listOf(hinge)))!!
        assertEquals(Rect(420f, 0f, 820f, 900f), regions.content)
    }

    @Test fun flatAndOutOfWindowHingesDoNotSplitTheWindow() {
        val flat = HingeInfo(Rect(400f, 0f, 400f, 900f), true, true, false, false)
        assertNull(hingeRegions(800f, 900f, Offset.Zero, Posture(hingeList = listOf(flat))))
        val outside = HingeInfo(Rect(1000f, 0f, 1024f, 900f), false, true, true, true)
        assertNull(hingeRegions(800f, 900f, Offset.Zero, Posture(hingeList = listOf(outside))))
    }
}
