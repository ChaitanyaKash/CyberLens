package io.github.chaitanyakash.cyberlens

import androidx.compose.material3.adaptive.Posture
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect

internal val LocalPostureOverride = staticCompositionLocalOf<Posture?> { null }

internal data class HingeRegions(
    val content: Rect,
    val navigation: Rect,
    val sheet: Rect,
    val vertical: Boolean,
)

/** WindowManager reports hinge bounds in window pixels, including zero-width creases. */
internal fun hingeRegions(width: Float, height: Float, origin: Offset, posture: Posture): HingeRegions? {
    if (width <= 0f || height <= 0f) return null
    val hinge = posture.hingeList.firstOrNull { hinge ->
        val b = hinge.bounds.translate(-origin.x, -origin.y)
        (hinge.isSeparating || hinge.isOccluding) && if (hinge.isVertical) {
            b.left > 0f && b.right < width && b.bottom > 0f && b.top < height
        } else {
            b.top > 0f && b.bottom < height && b.right > 0f && b.left < width
        }
    } ?: return null
    val bounds = hinge.bounds.translate(-origin.x, -origin.y)
    return if (hinge.isVertical) {
        val left = Rect(0f, 0f, bounds.left, height)
        val right = Rect(bounds.right, 0f, width, height)
        HingeRegions(content = right, navigation = left, sheet = right, vertical = true)
    } else {
        val top = Rect(0f, 0f, width, bounds.top)
        val bottom = Rect(0f, bounds.bottom, width, height)
        HingeRegions(content = top, navigation = bottom, sheet = bottom, vertical = false)
    }
}
