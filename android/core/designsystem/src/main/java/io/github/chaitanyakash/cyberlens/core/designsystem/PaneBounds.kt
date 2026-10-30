package io.github.chaitanyakash.cyberlens.core.designsystem

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.layout
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.unit.Constraints
import kotlin.math.roundToInt

/** The continuous window area in which a modal can place interactive content. */
val LocalModalPaneBounds = staticCompositionLocalOf<Rect?> { null }

@Composable
internal fun CLModalPane(content: @Composable BoxScope.() -> Unit) {
    val pane = LocalModalPaneBounds.current
    var origin by remember { mutableStateOf(Offset.Zero) }
    Box(Modifier.fillMaxSize().onGloballyPositioned { origin = it.positionInWindow() }
        .layout { measurable, constraints ->
            val width = constraints.maxWidth
            val height = constraints.maxHeight
            val left = pane?.let { (it.left - origin.x).roundToInt().coerceIn(0, width) } ?: 0
            val top = pane?.let { (it.top - origin.y).roundToInt().coerceIn(0, height) } ?: 0
            val right = pane?.let { (it.right - origin.x).roundToInt().coerceIn(left, width) } ?: width
            val bottom = pane?.let { (it.bottom - origin.y).roundToInt().coerceIn(top, height) } ?: height
            val child = measurable.measure(Constraints.fixed(right - left, bottom - top))
            layout(width, height) { child.place(left, top) }
        }.clipToBounds(), content = content)
}
